/*-----------------------Includes---------------------------------------*/
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/rtsp/gstrtsptransport.h>
#include <gst/rtp/gstrtcpbuffer.h>
#include <glib.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "cuda_runtime_api.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvds_version.h"
#include <future>
#include "../common/includes/MJPEGWriter.h"
#include <unistd.h>
#include <cstdlib>
#include <thread>  
#include "../common/includes/main.h"
#include <typeinfo>
#include <fstream>

#include <json.hpp>
using json = nlohmann::ordered_json;

#include <curlpp/cURLpp.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Easy.hpp>

using namespace curlpp::options;

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/md5.h>

#include <filesystem>
namespace fs = std::filesystem;

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>

/*-----------------------------------------------------------------------*/
/*-----------------------Models---------------------------------------*/
#define INFER_PGIE_CONFIG_FILE "/phoenix/configs/peoplenet/config_infer_primary_peoplenet.txt"
#define INFER_SGIE1_CONFIG_FILE "/phoenix/configs/garbage/config_infer_primary_garbage.txt"
/*-----------------------------------------------------------------------*/


/*-----------------------Constantes---------------------------------------*/
#define TRACKER  /* To enable tracker remove comments*/
//#define ANALYTICS  /* To enable anaylytics remove comments*/
#define CONFIDENCE

#define NVINFER_PLUGIN "nvinfer"
#define MUXER_BATCH_TIMEOUT_USEC 40000
#define MEMORY_FEATURES "memory:NVMM"

// #define PGIE_CLASS_ID_0 0
// #define PGIE_CLASS_ID_1 1
// #define PGIE_CLASS_ID_2 2
// #define PGIE_CLASS_ID_3 3

#define PGIE_COMPONENT_ID 1
#define SGIE_COMPONENT_ID 2

#define PERSON_ID 0 //IN PGIE
#define HELMET_ID 0 //IN SGIE1
#define VEST_ID 1 //IN SGIE1

#define DROP_FRAME_INTERVAL 8

#define PGIE_DETECTED_CLASS_NUM 2
#define MAX_SINK_BINS (1024)
#define MAX_INSTANCES 128
#define MAX_DISPLAY_LEN 64
#if (defined TRACKER)
  #define CHECK_ERROR(error) \
      if (error) { \
          g_printerr ("Error while parsing config file: %s\n", error->message); \
          goto done; \
      }
#endif
/*-----------------------------------------------------------------------*/
/*-----------------------ANALYTICS---------------------------------------*/
#if (defined ANALYTICS)
  #define NVDSANALYTICS_CONFIG_FILE "/phoenix/configs/analytics/config_nvdsanalytics.txt"
#endif
/*-----------------------------------------------------------------------*/
/*-----------------------Tracker---------------------------------------*/
#if (defined TRACKER)
  #define TRACKER_CONFIG_FILE "/phoenix/configs/tracker/tracker_config.txt"
  #define CONFIG_GPU_ID "gpu-id"
  #define CONFIG_GROUP_TRACKER "tracker"
  #define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
  #define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
  #define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
  #define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
  #define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#endif
/*-----------------------------------------------------------------------*/
/*-----------------------CLIENT API---------------------------------------*/
string api_url = std::getenv("API_URL");
string api_get_token_url = std::getenv("API_GET_TOKEN_URL");
string client_secret = std::getenv("CLIENT_SECRET");
string scope = std::getenv("SCOPE");
//string client_id = std::getenv("CLIENT_ID"); 
string client_id = "smartwaste.boitier1"; 
string grant_type = std::getenv("GRANT_TYPE");
string location_url = std::getenv("LOCATION_URL");
/*-----------------------------------------------------------------------*/
/*-----------------------RTSP OUT MAPPING--------------------------------*/
string MAPPING= std::getenv("RTSP_OUT_MAPPING");
/*-----------------------------------------------------------------------*/
/*-----------------------OUTPUT STREAM------------------------------------*/
//int MUXER_OUTPUT_WIDTH = std::stoi(std::getenv("OUTPUT_WIDTH"));
//int MUXER_OUTPUT_HEIGHT = std::stoi(std::getenv("OUTPUT_HIGHT"));
//int TILED_OUTPUT_WIDTH = std::stoi(std::getenv("OUTPUT_WIDTH"));
//int TILED_OUTPUT_HEIGHT = std::stoi(std::getenv("OUTPUT_HIGHT"));
int MUXER_OUTPUT_WIDTH = 1280;
int MUXER_OUTPUT_HEIGHT = 720;
int TILED_OUTPUT_WIDTH = 1280;
int TILED_OUTPUT_HEIGHT = 720;
/*-----------------------------------------------------------------------*/
/*-----------------------Variables---------------------------------------*/
AppCtx *appCtx[MAX_INSTANCES];
GMainLoop *loop = NULL;

const gchar pgie_classes_str[PGIE_DETECTED_CLASS_NUM][32] =
    { "helmet", 
      "vest",
    };

unsigned int nvds_lib_major_version = NVDS_VERSION_MAJOR;
unsigned int nvds_lib_minor_version = NVDS_VERSION_MINOR;
static GstRTSPServer *server [MAX_SINK_BINS];
static guint server_count = 0;
static GMutex server_cnt_lock;
guint num_sources = 0;

vector<float> tops;
vector<float> lefts;
vector<float> widths;
vector<float> heights;
vector<int> counts;
vector<int> life;
vector<bool> alerts;
vector<time_t> objects_init_life; 
vector<int> objects_life;

gint frame_number = 0;
int seconds = 0;
int token_expires_at;
int alert_thresh = 50 / (DROP_FRAME_INTERVAL + 1);
int initial_life= 50 / (DROP_FRAME_INTERVAL + 1); //Number of frames that we wait to see similar object
int video_duration;
int start_video, end_video; //Start recording of total video file
int video_segment_duration = 300; //15 min
int n_MB = 5;
int display_obj_sec = 0;
int offset_tz;
long int obj_alert_lifetime = 180; // After 100 seconds, alert is sent
long int obj_lifetime = 30; // Number of seconds we wait to see similar object -> if it doesn't happen ; object is killed 
long int garbage_lifetime = 10800; // Number of seconds we wait before killing detected garbage (vehicle occlusion, ...)
double ratio_width; // we can get rtsp in ; out is a parameter -> give ratio value in .env
double ratio_height;
//double mux_out_width = MUXER_OUTPUT_WIDTH;
//double mux_out_height = MUXER_OUTPUT_HEIGHT;
double mux_out_width = MUXER_OUTPUT_WIDTH;
double mux_out_height = MUXER_OUTPUT_HEIGHT;



string recording_file_name, next_recording_file_name;
string rtsp_in;
string webdav_url = "192.168.40.40:8081/alerts/";
string api_token;
string video_id;
string trigger;
string time_zone;

static gboolean install_mux_eosmonitor_probe = FALSE;
bool timer_updateInfos = true;
bool video_sender = false;
bool rtsp_lost = true; // if rtsp is lost, we have to check if resolution is still the same

json access_token;
json params;

// log message to file
auto max_size = 1048576 * 15;
auto max_files = 3;
auto info_logger = spdlog::rotating_logger_mt("App Info", "/phoenix/logs/info.log", max_size, max_files);
auto debug_logger = spdlog::rotating_logger_mt("App Debug", "/phoenix/logs/debug.log", max_size, max_files);

/*-----------------------------------------------------------------------*/

bool cmp(pair<string, int>& a,
         pair<string, int>& b)
{
    return a.second > b.second;
}

int datetimeToTimestamp(string path_tmp, string path)
{
  string delimiter = path + "/";
  path_tmp.erase(0, path_tmp.find(delimiter) + delimiter.length());
  delimiter = ".ts";
  path_tmp = path_tmp.substr(0, path_tmp.find(delimiter));
  delimiter = "-";
  int year = stoi(path_tmp.substr(0, path_tmp.find(delimiter)));
  path_tmp.erase(0, path_tmp.find(delimiter) + delimiter.length());
  int month = stoi(path_tmp.substr(0, path_tmp.find(delimiter)));
  path_tmp.erase(0, path_tmp.find(delimiter) + delimiter.length());
  delimiter = "_";
  int day = stoi(path_tmp.substr(0, path_tmp.find(delimiter)));
  path_tmp.erase(0, path_tmp.find(delimiter) + delimiter.length());
  

  delimiter = "-";
  int hours = stoi(path_tmp.substr(0, path_tmp.find(delimiter)));
  path_tmp.erase(0, path_tmp.find(delimiter) + delimiter.length());
  int minutes = stoi(path_tmp.substr(0, path_tmp.find(delimiter)));
  path_tmp.erase(0, path_tmp.find(delimiter) + delimiter.length());
  int seconds = stoi(path_tmp.substr(0, path_tmp.find(delimiter)));

  // cout << "Year: " << year << " - Month: " << month << " - Day: " << day << " - Hours: " << hours << " - Minutes: " << minutes << " - Seconds: " << seconds << endl;
  
  struct tm  tm;
  time_t rawtime;
  time ( &rawtime );
  tm = *localtime ( &rawtime );
  tm.tm_year = year - 1900;
  tm.tm_mon = month - 1;
  tm.tm_mday = day;
  tm.tm_hour = hours + offset_tz;
  tm.tm_min = minutes;
  tm.tm_sec = seconds;
  long int ts = mktime(&tm);

  // cout << "Timestamp: " << ts << endl;

  return ts;
}
double start_time=((double)clock())/CLOCKS_PER_SEC;;

double calc_fps()
{
  double end_time , elapsed_time,current_fps;
  end_time=((double)clock())/CLOCKS_PER_SEC;
  elapsed_time = end_time - start_time;
  current_fps = 1.0/float(elapsed_time);
  start_time=end_time;
  //str_fps = “%.1f”%(current_fps)
  return current_fps;
}

// get value from system command
std::string exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}

void getParams()
{
  ifstream i("./parameters.json");
  i >> params;

  trigger = params["parameters"]["trigger"];
  std::transform(trigger.begin(), trigger.end(), trigger.begin(), [](unsigned char c){ return std::tolower(c); });
  video_duration = params["parameters"]["duration"];
}



string hexStr(unsigned char* data)
{
     stringstream ss;
     ss << hex;

     for( int i(0) ; i < MD5_DIGEST_LENGTH; ++i )
         ss << setw(2) << setfill('0') << (int)data[i];

     return ss.str();
}

// Get the size of the file by its file descriptor
unsigned long get_size_by_fd(int fd) {
    struct stat statbuf;
    if(fstat(fd, &statbuf) < 0) exit(-1);
    return statbuf.st_size;
}

//https://softwarerecs.stackexchange.com/questions/49081/recommend-c-library-to-split-a-file-into-chunks-and-merge-it-back
// CUTTING INTO BINARY CHUNKS
const int size1MB = 1024 * 1024;

// CUTTING INTO BINARY CHUNKS
std::unique_ptr<std::ofstream> createChunkFile(std::vector<std::string>& vecFilenames) {
    std::stringstream filename;
    filename << "chunks_temp/chunk" << (vecFilenames.size() + 1) << ".txt";
    vecFilenames.push_back(filename.str());
    return std::make_unique<std::ofstream>(filename.str(), std::ios::trunc);
}

// CUTTING INTO BINARY CHUNKS
void split(std::istream& inStream, int nMegaBytesPerChunk, std::vector<std::string>& vecFilenames) {

    std::unique_ptr<char[]> buffer(new char[size1MB]);
    int nCurrentMegaBytes = 0;

    std::unique_ptr<std::ostream> pOutStream = createChunkFile(vecFilenames);

    while (!inStream.eof()) {
        inStream.read(buffer.get(), size1MB);
        pOutStream->write(buffer.get(), inStream.gcount());
        ++nCurrentMegaBytes;
        if (nCurrentMegaBytes >= nMegaBytesPerChunk) {
            pOutStream = createChunkFile(vecFilenames);
            nCurrentMegaBytes = 0;
        }
    }
}

// CUTTING INTO BINARY CHUNKS

void join(std::vector<std::string>& vecFilenames, std::ostream& outStream) {
    for (int n = 0; n < vecFilenames.size(); ++n) {
        std::ifstream ifs(vecFilenames[n]);
        outStream << ifs.rdbuf();
    }
}

// Check if file exists before splitting
inline bool exists_test (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

std::fstream& GotoLine(std::fstream& file, unsigned int num){
    file.seekg(std::ios::beg);
    for(int i=0; i < num - 1; ++i){
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    return file;
}


void sendAlert(int top, int left, int width,  int height)
{

  time_t now_time_t;
  time(&now_time_t);
  long int now = static_cast<long int> (now_time_t);

  string list_name;

  string uuid = exec("uuidgen"); 
  uuid.erase(std::remove(uuid.begin(), uuid.end(), '\n'), uuid.end());

  info_logger->info("ALERT OBJECT = " + uuid + " -> video duration = " + to_string(video_duration) + " seconds");

  cout << "=====================================================" << endl;
  cout << "ALERT OBJECT at " << now << endl;
  cout << "Video will be send, Total duration: " << video_duration << endl;
  cout << "UUID = " << uuid << endl;
  
  //int now = time(0);

  int begin_recording, end_recording, sleep_time;


  if(trigger=="beginning")
  {
    begin_recording = now - obj_alert_lifetime;
    end_recording = now + video_duration - obj_alert_lifetime;
    sleep_time = video_duration + 10;
  }
  else if(trigger=="end")
  {
    begin_recording = now - video_duration - obj_alert_lifetime;
    end_recording = now - obj_alert_lifetime;
    sleep_time = 0.1; //just sleep 0.1s 
  }
  else{ //always trigger=center if no other match, avoid error
    begin_recording = now - video_duration/2 - obj_alert_lifetime;
    end_recording = now + video_duration/2 - obj_alert_lifetime;
    sleep_time = video_duration/2 + 10;
  }


  sleep(sleep_time); //Record enough before cutting the video

  // Find video
  // TEST to list file, should be moved to recording part
  string path = "/phoenix/media/records";
  std::map<std::string, int> records {}; 
  string path_tmp;

  for (const auto & entry : fs::directory_iterator(path))
  {
    path_tmp = entry.path();
    if(path_tmp.find(".ts") != string::npos)
    {
      // map : key = path and value = timestamp
      records[path_tmp] = datetimeToTimestamp(path_tmp, path);
      // cout << path_tmp << endl;
    }
  }

  //Sort map by timestamp value
  // Declare vector of pairs
  vector<pair<string, int> > A;

  // Copy key-value pair from Map
  // to vector of pairs
  for (auto& it : records) {
      A.push_back(it);
  }

  // Sort using comparator function
  sort(A.begin(), A.end(), cmp); // Sort from newest to oldest

  // Print the sorted value
  for (auto& it : A) {
    //info_logger->info("FIRST=" + it.first);
    //info_logger->info("SECOND= " + it.second);
      //cout << it.first << ' '
            //<< it.second << endl;
  }

  std::cout << "mymap.size() is " << records.size() << '\n';
  int count = 0;


    




  for (auto elem : A) {
      std::cout << elem.first << " = " << elem.second << "; compared with " << begin_recording << endl;

      if (elem.second < begin_recording)
      {
        //cout << "Video containing recording is: " << elem.first << endl;
        info_logger->info("Video containing recording is: " + elem.first);

        info_logger->info("HERE0");

        recording_file_name = elem.first;
        start_video = elem.second;
        //break; //Take the oldest one that meets the criterion

        end_video = start_video + video_segment_duration;

        //cout << "End recording: " << end_recording << endl;
        //cout << "End video: " << end_video << endl;

        info_logger->info("HERE1");

        if(end_recording < end_video)
        {
          break; //The video is found
        }
        else
        {
          if(count>0) //Not the newest video --> take current video and next one in time
          {
            info_logger->info("HERE2");
            //cout << "!!!!!!!!!!!We will concatenate two videos !!" << endl;
            ofstream myfile;
            list_name = uuid + ".txt";
            myfile.open(list_name);
            // myfile << "Writing this to a file.\n";
            myfile << "file '" << recording_file_name << "'" << endl;
            myfile << "file '" << next_recording_file_name << "'" << endl;
            myfile.close();
             
            //info_logger->info("RECORD FN = " + recording_file_name);
            //info_logger->info("NEXT RECORD FN = " + next_recording_file_name);

            string command;
            command.append("ffmpeg -f concat -safe 0 -i " + list_name + " -c copy " + uuid + ".ts");
            system(command.c_str());

            info_logger->info("HERE3");

            recording_file_name = uuid + ".ts";
            break;
          }
          else
          {
            //Should not happen because the sleep was before
            cerr << "This shouldn't happen, the alert video is merely compromised\n";
          }
        }
        // Need to check if a second video is needed
            // First case: The selected video is not the newest and we are too close to the end of the video
            // Second case: The selected video is the newest but the video will end before the requested duration
      }
      next_recording_file_name = elem.first;
      count ++;
  }
  

  //info_logger->info("BEGIN " + begin_recording);
  //info_logger->info("END " + end_recording);
  
  //cout << "Start recording at: " << begin_recording << endl;
  //cout << "End recording at: " << end_recording << endl;
  //cout << "=====================================================" << endl;

  info_logger->info("HERE4");

  // 1. Record video
  int relative_begin = begin_recording - start_video;
  //int relative_end = end_recording - start_video;



  string file_name = "alert_" + uuid + ".mp4";

  string webdav_alert_url = webdav_url + file_name;


  //cout << "FILE SHOW = " << file_name << endl;
  

  string command;
  //command.append("(sleep ");
  // command.append(to_string(video_duration)); //Be sure to record enough without taking
  // command.append(" && ");
  command.append("ffmpeg -hide_banner -loglevel panic -i ");
  command.append(recording_file_name);
  command.append(" -ss ");
  command.append(to_string(relative_begin));
  command.append(" -t ");
  command.append(to_string(video_duration));
  command.append(" -vcodec copy -preset ultrafast -an /phoenix/media/alerts/");
  command.append(file_name);
  system(command.c_str());

  if(recording_file_name == uuid + ".ts")
  {
    remove(recording_file_name.c_str());
    remove(list_name.c_str());
  }

  info_logger->info("HERE5");

  
  // Envoi de la miniature de l'alerte

  string thumbnail_name = "alert_" + uuid  + ".jpg";

  // wait video is created before getting last frame
  bool wait = true;
  string path_vid = "/phoenix/media/alerts/" + file_name; 
  while(wait){
    if (exists_test(path_vid.c_str()) == 1){
      wait = false;
    }
  }

  command = "";
  command.append("ffmpeg -loglevel panic -sseof -3 -i /phoenix/media/alerts/");
  command.append(file_name);
  command.append(" -update 1 -q:v 1 /phoenix/media/alerts/");
  command.append(thumbnail_name);
  system(command.c_str());

  // wait frame is created before drawing bounding box
  wait = true; 
  string path_thum = "/phoenix/media/alerts/" + thumbnail_name; 
  while(wait){
    if (exists_test(path_thum.c_str()) == 1){
      wait = false;
    }
  }

  cv::Mat thumb_im;
  thumb_im = imread(path_thum);

  // define rect
  Point p1(int(left * ratio_width), int(top * ratio_height));
  Point p2(int((left + width) * ratio_width), int((top + height) * ratio_height));

  int thickness = 2;

  // draw rect
  cv::rectangle(thumb_im, p1, p2, Scalar(0, 255, 0), thickness, LINE_8);

  vector<int> p(2);
  p[0] = CV_IMWRITE_JPEG_QUALITY;
  p[1] = 70; // compression factor

  imwrite(path_thum, thumb_im, p);  
  
  // 2. Send alert to API

  // Compute MD5 sum of the file
  int file_descript;
  unsigned long file_size;
  char* file_buffer;

  unsigned char result[MD5_DIGEST_LENGTH];
  
  //printf("using file:\t%s\n", argv[1]);

  file_descript = open(path_vid.c_str(), O_RDONLY);
  if(file_descript < 0) exit(-1);

  file_size = get_size_by_fd(file_descript);
  //printf("file size:\t%lu\n", file_size);

  file_buffer = static_cast<char*>(mmap(0, file_size, PROT_READ, MAP_SHARED, file_descript, 0));
  MD5((unsigned char*) file_buffer, file_size, result);
  munmap(file_buffer, file_size); 

  //print_md5_sum(result);
  string md5_sum = hexStr(result);
  //cout << "md5_sum string: " << md5_sum << endl;
  //printf("  %s\n", file_name);

  // Get file size in bytes -> http://www.cplusplus.com/doc/tutorial/files/
  streampos begin,end;
  ifstream myfile (path_vid.c_str(), ios::binary);
  begin = myfile.tellg();
  myfile.seekg (0, ios::end);
  end = myfile.tellg();
  myfile.close();
  int file_size_bytes = (end-begin);
  //cout << "size is: " << (end-begin) << " bytes.\n";

  // Create json alert
  json alert;
  json send_alert_result;

  char buf[sizeof "2011-10-08T07:07:09Z"];
  strftime(buf, sizeof buf, "%FT%TZ", gmtime(&now_time_t));

  //Get location
  string latitude, longitude;
  fstream file("/phoenix/bin/gps.txt");
  GotoLine(file, 1);
  file >> latitude;
  GotoLine(file, 2);
  file >> longitude;

  int chunkSize = n_MB*1048576;

  alert["externalId"] = uuid; //timestamp begin recording is alert ID
  //alert["externalId"] = "alert_" + to_string(begin_recording) + "_" + uuid;
  alert["date"] = buf;
  alert["location"] = latitude + "," + longitude;
  alert["status"] = nullptr; // to define later
  alert["medias"][0]["externalId"] = uuid; //same id because only one media
  //alert["medias"][0]["externalId"] = "alert_" + to_string(begin_recording) + "_" + uuid;
  alert["medias"][0]["type"] = "Video";
  alert["medias"][0]["uploadSession"]["fileSize"] = file_size_bytes;
  alert["medias"][0]["uploadSession"]["fileExtension"] = "mp4";
  alert["medias"][0]["uploadSession"]["checkSum"] = md5_sum;
  alert["medias"][0]["uploadSession"]["chunkSize"] = chunkSize; // 1 Byte

  string alert_to_send = alert.dump();

  // write JSON to file
  string json_name = "alert_" + uuid  + ".json";
  string path_json = "/phoenix/media/alerts/" + json_name; 
  std::ofstream o(path_json);
  o << std::setw(4) << alert << std::endl;

  //cout << "ALERT JSON = " << alert_to_send << endl;

  if (file_size_bytes != 0){

    string command_email = "";
    command_email.append("echo 'A waste has been detected. You will find an attached image of the detected waste. A video is available at this address:" + webdav_alert_url + "' | mutt -s 'Waste Detected' -a ");
    command_email.append(path_thum);
    command_email.append(" -c simon@phoenix-ai.com");

    //system(command_email.c_str());

    //info_logger->info(uuid + " -> EMAIL SENT");
    cout << "EMAIL SENT" << endl;

  }

}


void checkInputStreamResol(){

  info_logger->info("WARNING : calculating new RTSP ratio");
  cout << "WARNING : calculating new RTSP ratio" << endl;

  string command_resol_height = "";
  string command_resol_width = "";

  command_resol_width.append("ffmpeg -i '");
  command_resol_width.append(rtsp_in);
  command_resol_width.append("' 2>&1 | grep Video: | grep -Po '\\d{3,5}x\\d{3,5}' | cut -d'x' -f1");

  command_resol_height.append("ffmpeg -i '");
  command_resol_height.append(rtsp_in);
  command_resol_height.append("' 2>&1 | grep Video: | grep -Po '\\d{3,5}x\\d{3,5}' | cut -d'x' -f2");

  //string resol_width = exec(command_resol_width.c_str());
  //string resol_height = exec(command_resol_height.c_str());
  string resol_width = "1280";
  string resol_height = "720";
  resol_width.erase(std::remove(resol_width.begin(), resol_width.end(), '\n'), resol_width.end());
  resol_height.erase(std::remove(resol_height.begin(), resol_height.end(), '\n'), resol_height.end());
  
  if(resol_width.empty()){
      debug_logger->error("ERROR : NO STREAM -> cannot calculate new RTSP ratio");
      cout << "ERROR : NO STREAM -> cannot calculate new RTSP ratio" << endl;
  }

  info_logger->info("RTSP RESOLUTION : INPUT WIDTH = " + resol_width + " AND INPUT HEIGHT = " + resol_height);

  ratio_width = stod(resol_width) / mux_out_width; 
  ratio_height = stod(resol_height) / mux_out_height;
  
  info_logger->info("RTSP NEW RATIO = " + to_string(ratio_width) + " AND " + to_string(ratio_height));
  cout << "NEW RTSP RATIO = " << ratio_width << " and " << ratio_height << endl;
}



// faire tout le traitement de ce qui vient de deepstream -> fonction appelée continuellement en parcourant les batchs de frames (batch de 1 frame)
static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint count_0 = 0;
  guint count_1 = 0;
  guint count_2 = 0;
  guint count_3 = 0;
  guint truck_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_cls = NULL;
  NvDsDisplayMeta *display_meta3 = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);



  if(rtsp_lost){
    thread t5(checkInputStreamResol);
    t5.detach();
    rtsp_lost = false;
  }



  // Check if token still valid
  if(frame_number%100 == 0)
  {
    // 2. Retrieve params from JSON
    getParams();
  }
  
 
  // peu importe la taille du batch, ici, c'est tjrs frame par frame
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;l_frame = l_frame->next) 
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) 
    {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      display_meta3 = nvds_acquire_display_meta_from_pool (batch_meta);
      NvOSD_TextParams *txt_params3 = &display_meta3->text_params[0];
      for (l_cls = obj_meta->classifier_meta_list; l_cls != NULL; l_cls = l_cls->next) 
      {
        NvDsClassifierMeta *cls_meta = (NvDsClassifierMeta *) (l_cls->data);
      }
    
      // Person bounding boxes
      if(obj_meta->unique_component_id==PGIE_COMPONENT_ID && obj_meta->class_id==PERSON_ID)
      {
        // To display bounding boxes for people
        obj_meta->rect_params.border_color = {0.0, 1.0, 0.0, 1.0}; //green
        obj_meta->rect_params.border_width = 0;
        obj_meta->text_params.font_params.font_size = 0;
        //cout << "Person detected" << endl;
      }
      else if(obj_meta->unique_component_id==SGIE_COMPONENT_ID && obj_meta->class_id==VEST_ID)
      {
        obj_meta->rect_params.border_color = {26.0/255, 94.0/255, 230.0/255, 1.0}; //purple
        obj_meta->text_params.font_params.font_size = 0;
        // cout <<
      }
      else if(obj_meta->unique_component_id==SGIE_COMPONENT_ID && obj_meta->class_id==HELMET_ID)
      {
        obj_meta->rect_params.border_color = {211.0/255, 3.0/255, 252.0/255, 1.0}; //purple
        obj_meta->text_params.font_params.font_size = 0;
        // cout << "Garbage detected" << endl;

        bool similar = false;
        for(int i=0; i<tops.size(); i++) //Process all stored objects
        {
          
          //- checker si c'est les mêmes bounding boxes ou pas -> si counts = 200/9 (=22) -> alert!
          if(abs(obj_meta->rect_params.left - lefts[i]) < 0.035 * MUXER_OUTPUT_WIDTH &&
            abs(obj_meta->rect_params.top - tops[i]) < 0.035 * MUXER_OUTPUT_HEIGHT &&
            abs(obj_meta->rect_params.width - widths[i]) < 0.035 * MUXER_OUTPUT_WIDTH &&
            abs(obj_meta->rect_params.height - heights[i]) < 0.035 * MUXER_OUTPUT_HEIGHT)
          {
            // cout << "Similar object at position " << i << endl;
            counts[i]++;
      
            // cout << "tops[i]: " << tops[i] << endl;
            // cout << "lefts[i]: " << lefts[i] << endl;
            // cout << "widths[i]: " << widths[i] << endl;
            // cout << "heights[i]: " << heights[i] << endl;
            // cout << "Left diff: " << abs(obj_meta->rect_params.left - lefts[i]) << endl;
            // cout << "Left tol: " << 0.1 * MUXER_OUTPUT_WIDTH << endl;
            // cout << "Top diff: " << abs(obj_meta->rect_params.top - tops[i]) << endl;
            // cout << "Top tol: " << 0.1 * MUXER_OUTPUT_HEIGHT << endl;
            // cout << "Width diff: " << abs(obj_meta->rect_params.width - widths[i]) << endl;
            // cout << "Width tol: " << 0.1 * MUXER_OUTPUT_WIDTH << endl;
            // cout << "Height diff: " << abs(obj_meta->rect_params.height - heights[i]) << endl;
            // cout << "Height tol: " << 0.1 * MUXER_OUTPUT_HEIGHT << endl;
            // cout << "Counts: ";
            // int max = int(*max_element(counts.begin(), counts.end()));
            // if (max%100==0)
            // {
              // for(int j=0; j<counts.size(); j++)
              //   cout << counts[j] << ' ';
              // cout << endl;
            // }
            

            //Update mean position
            tops[i] = (1-1.0/counts[i])*tops[i] + 1.0/counts[i] * obj_meta->rect_params.top;
            lefts[i] = (1-1.0/counts[i])*lefts[i] + 1.0/counts[i] * obj_meta->rect_params.left;
            widths[i] = (1-1.0/counts[i])*widths[i] + 1.0/counts[i] * obj_meta->rect_params.width;
            heights[i] = (1-1.0/counts[i])*heights[i] + 1.0/counts[i] * obj_meta->rect_params.height;

            similar = true;

            if (time(NULL) % 60 == 0 && time(NULL) != display_obj_sec){
              info_logger->info("INFO : DISPLAYING EXISTING OBJECTS : ");
              cout << "INFO : DISPLAYING EXISTING OBJECTS : " << endl;
              for (int j=0; j < objects_init_life.size(); j++){
                string x_center = to_string(int((lefts[j] + (widths[j]/2)) * ratio_width));
                string y_center = to_string(int((tops[j] - (heights[j] / 2)) * ratio_height));
                info_logger->info("OBJET " + to_string(j) + " -> center = (" + x_center + ";" + y_center + ") -> temps d'existance : " + to_string(time(NULL) - objects_init_life[j]));          
                cout << "OBJET " << j << " -> temps d'existance : " << time(NULL) - objects_init_life[j] << endl;
                display_obj_sec = time(NULL);
              }
            }
            
            if(!alerts[i] && time(NULL) - objects_init_life[i] >= obj_alert_lifetime)
            {
              alerts[i] = true; // to send alert only one time
              
              thread t1(sendAlert, tops[i], lefts[i], widths[i], heights[i]);
              t1.detach();
              // sendAlert();
              //sendAlert(alert.dump()); //Wait before sending alert !!

              //cout << "LIFE TIMER -> object life " << i << " = " << time(NULL) - objects_init_life[i] << " seconds" << endl;  

            }

            for (int j=0; j<objects_life.size(); j++)
            {
              if (i==j) // i = ième stored object
              {
                //life[j]=initial_life; //restore life
                objects_life[j] = time(NULL);
              }

              if(time(NULL) - objects_life[j] >= obj_lifetime && !alerts[j]) //end of live for elements that haven't been seen since obj_lifetime
              {
                string x_center = to_string(int((lefts[j] + (widths[j]/2)) * ratio_width));
                string y_center = to_string(int((tops[j] - (heights[j] / 2)) * ratio_height));
                info_logger->info("ERASING OBJECT -> center = (" + x_center + ";" + y_center + ")");
                cout << "ERASING OBJECT" << endl;
                counts.erase(counts.begin() + j);
                alerts.erase(alerts.begin() + j);
                tops.erase(tops.begin() + j);
                lefts.erase(lefts.begin() + j);
                widths.erase(widths.begin() + j);
                heights.erase(heights.begin() + j);
                objects_init_life.erase(objects_init_life.begin() + j);
                objects_life.erase(objects_life.begin() + j);
                j--;
              }

              if(time(NULL) - objects_life[j] >= garbage_lifetime && alerts[j]) //end of live for elements that haven't been seen since obj_lifetime
              {
                //life.erase(life.begin() + j);
                info_logger->info("ERASING OBJECT");
                cout << "ERASING OBJECT" << endl;
                counts.erase(counts.begin() + j);
                alerts.erase(alerts.begin() + j);
                tops.erase(tops.begin() + j);
                lefts.erase(lefts.begin() + j);
                widths.erase(widths.begin() + j);
                heights.erase(heights.begin() + j);
                objects_init_life.erase(objects_init_life.begin() + j);
                objects_life.erase(objects_life.begin() + j);
                j--;
              }
       
            }


            // cout << "tops: ";
            // for(int j=0; j<tops.size();j++)
            //   cout << tops[j] << ' ';
            // cout << endl;

            // cout << "life: ";
            // for(int j=0; j<life.size();j++)
            //   cout << life[j] << ' ';
            // cout << endl;

            // cout << "------------------------------------------" << endl;

            
            break;
          }
        }
        if(!similar) // si pas similaire, on ajoute un nouvel objet
        {
          tops.push_back(obj_meta->rect_params.top);
          lefts.push_back(obj_meta->rect_params.left);
          widths.push_back(obj_meta->rect_params.width);
          heights.push_back(obj_meta->rect_params.height);
          counts.push_back(1);
          //life.push_back(initial_life);
          alerts.push_back(false);
          objects_init_life.push_back(time(NULL)); // get seconds since 01/1970
          objects_life.push_back(time(NULL));
        }
        
      }
      else{ // Don't show detection result
        obj_meta->rect_params.border_width = (unsigned int)0;
        obj_meta->text_params.set_bg_clr = (int)0; //false
        obj_meta->text_params.font_params.font_size = 0; //Remove text
      }


      // switch (obj_meta->class_id) 
      // {
       
      //   case PGIE_CLASS_ID_0:
      //      cout << "ID : " << obj_meta->object_id << endl;
      //     count_0++;
      //     num_rects++;
      //     txt_params3->text_bg_clr.red = 0.0;
      //     txt_params3->text_bg_clr.green = 0.0;
      //     txt_params3->text_bg_clr.blue = 0.0;
      //     break;
      //   case PGIE_CLASS_ID_1:
      //     count_1++;
      //     num_rects++;
      //     txt_params3->text_bg_clr.red = 0.5;
      //     txt_params3->text_bg_clr.green = 0.0;
      //     txt_params3->text_bg_clr.blue = 0.0;
      //     break;
      //   case PGIE_CLASS_ID_2:
      //     count_2++;
      //     num_rects++;
      //     txt_params3->text_bg_clr.red = 0.0;
      //     txt_params3->text_bg_clr.green = 0.5;
      //     txt_params3->text_bg_clr.blue = 0.0;
      //     break;
      //   case PGIE_CLASS_ID_3:
      //     count_3++;
      //     num_rects++;
      //     txt_params3->text_bg_clr.red = 1.0;
      //     txt_params3->text_bg_clr.green = 1.0;
      //     txt_params3->text_bg_clr.blue = 1.0;
      //     break;
        
      // }
      #if (defined CONFIDENCE)
        
        if(obj_meta->unique_component_id==SGIE_COMPONENT_ID && obj_meta->class_id==HELMET_ID || obj_meta->class_id==VEST_ID){

          NvOSD_RectParams rect_params = obj_meta->rect_params;
          float left = rect_params.left;
          float top = rect_params.top;
          float width = rect_params.width;
          float height = rect_params.height;
          float confidence = obj_meta->confidence;

          std::string result;
          char result_0[32];
          char result_1[32];
          display_meta3->num_labels = 1;
          txt_params3->display_text = (gchar *) g_malloc0 (MAX_DISPLAY_LEN);


          snprintf(result_0, 32, "%s ", pgie_classes_str[obj_meta->class_id]);
          snprintf(result_1, 32, "%.0f％", confidence * 100);


          result.append(result_0);
          result.append(result_1);

          offset = snprintf (
            txt_params3->display_text, MAX_DISPLAY_LEN, "%s", result.c_str()
          );
          /* Now set the offsets where the string should appear */
          txt_params3->x_offset = left;
          txt_params3->y_offset = top-40;

          /* Font , font-color and font-size */
          txt_params3->font_params.font_name = (gchar *) "Serif";
          txt_params3->font_params.font_size = 20;
          txt_params3->font_params.font_color.red = 0.0;
          txt_params3->font_params.font_color.green = 0.0;
          txt_params3->font_params.font_color.blue = 0.0;
          txt_params3->font_params.font_color.alpha = 1.0;

          /* Text background color */
          txt_params3->set_bg_clr = 1;
          txt_params3->text_bg_clr.red = 1.0;
          txt_params3->text_bg_clr.green = 1.0;
          txt_params3->text_bg_clr.blue = 1.0;
          txt_params3->text_bg_clr.alpha = 0.5;
        }

        
      #endif

      nvds_add_display_meta_to_frame (frame_meta, display_meta3);

    }
    // if (num_rects > 0) {
    //   g_print ("Frame Number = %d Number of objects = %d "
    //       "ID_0: %d ID_1: %d ID_2: %d ID_3: %d \n",
    //       frame_meta->frame_num, num_rects, count_0, count_1, count_2, count_3);  
    // }
  }
  //string fps_show_env=std::getenv("SHOW_FPS");
  string fps_show_env="NO";
  //string fps_show_env="YES";
  string to_compare_en="YES";
  if (fps_show_env.compare(to_compare_en) == 0)
  {
    g_print("FPS : %lf \n ", calc_fps());  
  }
  frame_number++;
  return GST_PAD_PROBE_OK;
}

gboolean link_element_to_streammux_sink_pad (GstElement *streammux, GstElement *elem,gint index)
{
  gboolean ret = FALSE;
  GstPad *mux_sink_pad = NULL;
  GstPad *src_pad = NULL;
  gchar pad_name[16];

  if (index >= 0) {
    g_snprintf (pad_name, 16, "sink_%u", index);
    pad_name[15] = '\0';
  } else {
    strcpy (pad_name, "sink_%u");
  }

  mux_sink_pad = gst_element_get_request_pad (streammux, pad_name);
  if (!mux_sink_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get sink pad from streammux");
    goto done;
  }

  src_pad = gst_element_get_static_pad (elem, "src");
  if (!src_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get src pad from '%s'",
                        GST_ELEMENT_NAME (elem));
    goto done;
  }

  if (gst_pad_link (src_pad, mux_sink_pad) != GST_PAD_LINK_OK) {
    NVGSTDS_ERR_MSG_V ("Failed to link '%s' and '%s'", GST_ELEMENT_NAME (streammux),
        GST_ELEMENT_NAME (elem));
    goto done;
  }

  ret = TRUE;

done:
  if (mux_sink_pad) {
    gst_object_unref (mux_sink_pad);
  }
  if (src_pad) {
    gst_object_unref (src_pad);
  }
  return ret;
}

gboolean
link_element_to_tee_src_pad (GstElement *tee, GstElement *sinkelem)
{
  gboolean ret = FALSE;
  GstPad *tee_src_pad = NULL;
  GstPad *sinkpad = NULL;
  GstPadTemplate *padtemplate = NULL;

  padtemplate = (GstPadTemplate *)
      gst_element_class_get_pad_template (GST_ELEMENT_GET_CLASS (tee),
      "src_%u");
  tee_src_pad = gst_element_request_pad (tee, padtemplate, NULL, NULL);
  if (!tee_src_pad) {
    NVGSTDS_ERR_MSG_V ("Failed to get src pad from tee");
    goto done;
  }

  sinkpad = gst_element_get_static_pad (sinkelem, "sink");
  if (!sinkpad) {
    NVGSTDS_ERR_MSG_V ("Failed to get sink pad from '%s'",
        GST_ELEMENT_NAME (sinkelem));
    goto done;
  }

  if (gst_pad_link (tee_src_pad, sinkpad) != GST_PAD_LINK_OK) {
    NVGSTDS_ERR_MSG_V ("Failed to link '%s' and '%s'", GST_ELEMENT_NAME (tee),
        GST_ELEMENT_NAME (sinkelem));
    goto done;
  }

  ret = TRUE;

done:
  if (tee_src_pad) {
    gst_object_unref (tee_src_pad);
  }
  if (sinkpad) {
    gst_object_unref (sinkpad);
  }
  return ret;
}

/**
 * Function called at regular interval when source bin is
 * changing state async. This function watches the state of
 * the source bin and sets it to PLAYING if the state of source
 * bin stops at PAUSED when changing state ASYNC.
 */
static gboolean
watch_source_async_state_change (gpointer data)
{
  NvDsSrcBin *src_bin = (NvDsSrcBin *) data;
  GstState state, pending;
  GstStateChangeReturn ret;

  ret = gst_element_get_state (src_bin->bin, &state, &pending, 0);

  // Bin is still changing state ASYNC. Wait for some more time.
  if (ret == GST_STATE_CHANGE_ASYNC)
    return TRUE;

  // Bin state change failed / failed to get state
  if (ret == GST_STATE_CHANGE_FAILURE) {
    src_bin->async_state_watch_running = FALSE;
    return FALSE;
  }

  // Bin successfully changed state to PLAYING. Stop watching state
  if (state == GST_STATE_PLAYING) {
    src_bin->reconfiguring = FALSE;
    src_bin->async_state_watch_running = FALSE;
    return FALSE;
  }

  // Bin has stopped ASYNC state change but has not gone into
  // PLAYING. Expliclity set state to PLAYING and keep watching
  // state
  gst_element_set_state (src_bin->bin, GST_STATE_PLAYING);

  return TRUE;
}





/**
 * Probe function to monitor data output from rtspsrc.
 */
static GstPadProbeReturn
rtspsrc_monitor_probe_func (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  NvDsSrcBin *bin = (NvDsSrcBin *) u_data;
  if (info->type & GST_PAD_PROBE_TYPE_BUFFER) {
    g_mutex_lock(&bin->bin_lock);
    gettimeofday (&bin->last_buffer_time, NULL);
    g_mutex_unlock(&bin->bin_lock);
  }
  return GST_PAD_PROBE_OK;
}

gboolean
reset_source_pipeline (gpointer data)
{
  NvDsSrcBin *src_bin = (NvDsSrcBin *) data;
  GstState state, pending;
  GstStateChangeReturn ret;

  g_mutex_lock(&src_bin->bin_lock);
  gettimeofday (&src_bin->last_buffer_time, NULL);
  gettimeofday (&src_bin->last_reconnect_time, NULL);
  g_mutex_unlock(&src_bin->bin_lock);

  if (gst_element_set_state (src_bin->bin,
          GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (src_bin->bin, "Can't set source bin to NULL");
    return FALSE;
  }
  NVGSTDS_INFO_MSG_V ("Resetting source %d", src_bin->bin_id);

  rtsp_lost = true;

  // GST_CAT_INFO (NVDS_APP, "Reset source pipeline %s %p\n,", __func__, src_bin);
  if (!gst_element_sync_state_with_parent (src_bin->bin)) {
    GST_ERROR_OBJECT (src_bin->bin, "Couldn't sync state with parent");
  }

  ret = gst_element_get_state (src_bin->bin, &state, &pending, 0);

  if (ret == GST_STATE_CHANGE_ASYNC || ret == GST_STATE_CHANGE_NO_PREROLL) {
    if (!src_bin->async_state_watch_running)
      g_timeout_add (20, watch_source_async_state_change, src_bin);
    src_bin->async_state_watch_running = TRUE;
    src_bin->reconfiguring = TRUE;
  } else if (ret == GST_STATE_CHANGE_SUCCESS && state == GST_STATE_PLAYING) {
    src_bin->reconfiguring = FALSE;
  }

  return FALSE;
}

/**
 * Function called at regular interval to check if NV_DS_SOURCE_RTSP type
 * source in the pipeline is down / disconnected. This function try to
 * reconnect the source by resetting that source pipeline.
 */
static gboolean
watch_source_status (gpointer data)
{
  NvDsSrcBin *src_bin = (NvDsSrcBin *) data;
  struct timeval current_time;
  gettimeofday (&current_time, NULL);
  static struct timeval last_reset_time_global = {0, 0};
  gdouble time_diff_msec_since_last_reset =
      1000.0 * (current_time.tv_sec - last_reset_time_global.tv_sec) +
      (current_time.tv_usec - last_reset_time_global.tv_usec) / 1000.0;

  if (src_bin->reconfiguring) {
    guint time_since_last_reconnect_sec =
        current_time.tv_sec - src_bin->last_reconnect_time.tv_sec;
    if (time_since_last_reconnect_sec >= SOURCE_RESET_INTERVAL_SEC) {
      if (time_diff_msec_since_last_reset > 3000) {
        last_reset_time_global = current_time;
        // source is still not up, reconfigure it again.
        reset_source_pipeline (src_bin);
      }
    }
  } else {
    gint time_since_last_buf_sec = 0;
    g_mutex_lock (&src_bin->bin_lock);
    if (src_bin->last_buffer_time.tv_sec != 0) {
      time_since_last_buf_sec =
          current_time.tv_sec - src_bin->last_buffer_time.tv_sec;
    }
    g_mutex_unlock (&src_bin->bin_lock);

    // Reset source bin if no buffers are received in the last
    // `rtsp_reconnect_interval_sec` seconds.
    if (src_bin->rtsp_reconnect_interval_sec > 0 &&
            time_since_last_buf_sec >= src_bin->rtsp_reconnect_interval_sec) {
      if (time_diff_msec_since_last_reset > 3000) {
        last_reset_time_global = current_time;

        NVGSTDS_WARN_MSG_V ("No data from source %d since last %u sec. Trying reconnection",
              src_bin->bin_id, time_since_last_buf_sec);
        reset_source_pipeline (src_bin);
      }
    }

  }
  return TRUE;
}

/**
 * callback function to receive messages from components
 * in the pipeline.
 */
static gboolean
bus_callback (GstBus * bus, GstMessage * message, gpointer data)
{
  AppCtx *appCtx = (AppCtx *) data;
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_INFO:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      gst_message_parse_info (message, &error, &debuginfo);
      g_printerr ("INFO from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }
      g_error_free (error);
      g_free (debuginfo);
      break;
    }
    case GST_MESSAGE_WARNING:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      gst_message_parse_warning (message, &error, &debuginfo);
      g_printerr ("WARNING from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }
      g_error_free (error);
      g_free (debuginfo);
      break;
    }
    case GST_MESSAGE_ERROR:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      guint i = 0;
      gst_message_parse_error (message, &error, &debuginfo);
      g_printerr ("ERROR from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }
      NvDsSrcParentBin *bin = &appCtx->pipeline.multi_src_bin;
      GstElement *msg_src_elem = (GstElement *) GST_MESSAGE_SRC (message);
      gboolean bin_found = FALSE;
      /* Find the source bin which generated the error. */
      while (msg_src_elem && !bin_found) {
        for (i = 0; i < bin->num_bins && !bin_found; i++) {
          printf("msg_src_elem: %s\n", gst_element_get_name(msg_src_elem));

          if (!bin->sub_bins[i].src_elem) {
            printf("bin->sub_bins[i].src_elem is empty\n");
            goto done;
          }
          if (bin->sub_bins[i].src_elem == msg_src_elem ||
                  bin->sub_bins[i].bin == msg_src_elem) {
            bin_found = TRUE;
            printf("bin to reset is found\n");

            break;
          }
        }
        msg_src_elem = GST_ELEMENT_PARENT (msg_src_elem);
      }

      if (i != bin->num_bins) {
        // Error from one of RTSP source.
        NvDsSrcBin *subBin = &bin->sub_bins[i];
        if (!subBin->reconfiguring ||
            g_strrstr(debuginfo, "500 (Internal Server Error)")) {
          subBin->reconfiguring = TRUE;
          printf("trying to call reset_source_pipeline\n");
          g_timeout_add (0, reset_source_pipeline, subBin);
        }
        g_error_free (error);
        g_free (debuginfo);
        return TRUE;
      }

    done:
      g_error_free (error);
      g_free (debuginfo);
      appCtx->return_value = -1;
      appCtx->quit = TRUE;
      break;
    }
    case GST_MESSAGE_STATE_CHANGED:{
      GstState oldstate, newstate;
      gst_message_parse_state_changed (message, &oldstate, &newstate, NULL);
      if (GST_ELEMENT (GST_MESSAGE_SRC (message)) == appCtx->pipeline.pipeline) {
        switch (newstate) {
          case GST_STATE_PLAYING:
            NVGSTDS_INFO_MSG_V ("Pipeline running\n");
            GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx->
                    pipeline.pipeline), GST_DEBUG_GRAPH_SHOW_ALL,
                "ds-app-playing");
            break;
          case GST_STATE_PAUSED:
            if (oldstate == GST_STATE_PLAYING) {
              NVGSTDS_INFO_MSG_V ("Pipeline paused\n");
            }
            break;
          case GST_STATE_READY:
            GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (appCtx->pipeline.
                    pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "ds-app-ready");
            if (oldstate == GST_STATE_NULL) {
              NVGSTDS_INFO_MSG_V ("Pipeline ready\n");
            } else {
              NVGSTDS_INFO_MSG_V ("Pipeline stopped\n");
            }
            break;
          case GST_STATE_NULL:
            g_mutex_lock (&appCtx->app_lock);
            g_cond_broadcast (&appCtx->app_cond);
            g_mutex_unlock (&appCtx->app_lock);
            break;
          default:
            break;
        }
      }
      break;
    }
    case GST_MESSAGE_EOS:{
      /*
       * In normal scenario, this would use g_main_loop_quit() to exit the
       * loop and release the resources. Since this application might be
       * running multiple pipelines through configuration files, it should wait
       * till all pipelines are done.
       */
      NVGSTDS_INFO_MSG_V ("Received EOS. Exiting ...\n");
      appCtx->quit = TRUE;
      return FALSE;
      break;
    }
    default:
      break;
  }
  return TRUE;
}


/* Returning FALSE from this callback will make rtspsrc ignore the stream.
 * Ignore audio and add the proper depay element based on codec. */
static gboolean
cb_rtspsrc_select_stream (GstElement *rtspsrc, guint num, GstCaps *caps,
        gpointer user_data)
{
  GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *media = gst_structure_get_string (str, "media");
  const gchar *encoding_name = gst_structure_get_string (str, "encoding-name");
  gchar elem_name[50];
  NvDsSrcBin *bin = (NvDsSrcBin *) user_data;
  gboolean ret = FALSE;

  gboolean is_video = (!g_strcmp0 (media, "video"));

  if (!is_video)
    return FALSE;

  /* Create and add depay element only if it is not created yet. */
  if (!bin->depay) {
    g_snprintf (elem_name, sizeof (elem_name), "depay_elem%d", bin->bin_id);

    /* Add the proper depay element based on codec. */
    if (!g_strcmp0 (encoding_name, "H264")) {
      bin->depay = gst_element_factory_make ("rtph264depay", elem_name);
      g_snprintf (elem_name, sizeof (elem_name), "h264parse_elem%d", bin->bin_id);
      bin->parser = gst_element_factory_make ("h264parse", elem_name);
    } else if (!g_strcmp0 (encoding_name, "H265")) {
      printf("passes rtph265depay\n");
      bin->depay = gst_element_factory_make ("rtph265depay", elem_name);
      g_snprintf (elem_name, sizeof (elem_name), "h265parse_elem%d", bin->bin_id);
      bin->parser = gst_element_factory_make ("h265parse", elem_name);
    } else {
      NVGSTDS_WARN_MSG_V ("%s not supported", encoding_name);
      return FALSE;
    }

    if (!bin->depay) {
      NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
      return FALSE;
    }
    gst_bin_add_many (GST_BIN (bin->bin), bin->depay, bin->parser, NULL);

    NVGSTDS_LINK_ELEMENT (bin->depay, bin->parser);
    NVGSTDS_LINK_ELEMENT (bin->parser, bin->tee_rtsp_pre_decode);

    if (!gst_element_sync_state_with_parent (bin->depay)) {
      NVGSTDS_ERR_MSG_V ("'%s' failed to sync state with parent", elem_name);
      return FALSE;
    }
    gst_element_sync_state_with_parent (bin->parser);
  }

  ret = TRUE;
done:
  return ret;
}

static void
cb_newpad2 (GstElement * decodebin, GstPad * pad, gpointer data)
{
  GstCaps *caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  gint input_width = 1280;
  gint input_height = 720;
  gint input_fps_n = 1;
  gint input_fps_d = 1;
  if (!strncmp (name, "video", 5)) {
    NvDsSrcBin *bin = (NvDsSrcBin *) data;
    GstPad *sinkpad = gst_element_get_static_pad (bin->cap_filter, "sink");
    if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {

      NVGSTDS_ERR_MSG_V ("Failed to link decodebin to pipeline");
    } else {
      gst_structure_get_int (str, "width", &input_width);
      gst_structure_get_int (str, "height", &input_height);
      gst_structure_get_fraction (str, "framerate", &input_fps_n, &input_fps_d);
    }
    gst_object_unref (sinkpad);
  }
  gst_caps_unref (caps);
}

static void
cb_newpad3 (GstElement * decodebin, GstPad * pad, gpointer data)
{
  GstCaps *caps = gst_pad_query_caps (pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  if (g_strrstr (name, "x-rtp")) {
    NvDsSrcBin *bin = (NvDsSrcBin *) data;
    GstPad *sinkpad = gst_element_get_static_pad (bin->depay, "sink");
    if (gst_pad_link (pad, sinkpad) != GST_PAD_LINK_OK) {

      NVGSTDS_ERR_MSG_V ("Failed to link depay loader to rtsp src");
    }
    gst_object_unref (sinkpad);
  }
  gst_caps_unref (caps);
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  if (!strncmp (name, "video", 5)) {
    if (gst_caps_features_contains (features, MEMORY_FEATURES)) {
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  NvDsSrcBin *bin = (NvDsSrcBin *) user_data;

  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if ((g_strrstr (name, "h264parse") == name) ||
      (g_strrstr (name, "h265parse") == name)) {
      g_object_set (object, "config-interval", -1, NULL);
  }
  if (g_strrstr (name, "fakesink") == name) {
      g_object_set (object, "enable-last-sample", FALSE, NULL);
  }
  if (g_strrstr (name, "nvcuvid") == name) {
    g_object_set (object, "gpu-id", 0, NULL);
  }
  if (g_strstr_len (name, -1, "omx") == name) {
    g_object_set (object, "skip-frames", 2, NULL);
    g_object_set (object, "disable-dvfs", TRUE, NULL);
  }
  if (g_strstr_len (name, -1, "nvjpegdec") == name) {
    g_object_set (object, "DeepStream", TRUE, NULL);
  }
  if (g_strstr_len (name, -1, "nvv4l2decoder") == name) {
#ifdef __aarch64__
    g_object_set (object, "enable-max-performance", TRUE, NULL);
#endif
    g_object_set(object, "drop-frame-interval", DROP_FRAME_INTERVAL, NULL);
  }
done:
  return;
}

static gboolean
create_rtsp_src_bin (guint index, gchar * location, NvDsSrcBin * bin)
{
  NvDsSRContext *ctx = NULL;
  gboolean ret = FALSE;
  gchar elem_name[50];
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;

  bin->rtsp_reconnect_interval_sec = 10;

  g_snprintf (elem_name, sizeof (elem_name), "src_elem%d", bin->bin_id);
  bin->src_elem = gst_element_factory_make ("rtspsrc", elem_name);
  if (!bin->src_elem) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  g_signal_connect (G_OBJECT(bin->src_elem), "select-stream",
                    G_CALLBACK(cb_rtspsrc_select_stream),
                    bin);

  g_object_set (G_OBJECT (bin->src_elem), "location", location, NULL);
  g_object_set (G_OBJECT (bin->src_elem), "latency", bin->latency, NULL);
  g_object_set (G_OBJECT (bin->src_elem), "drop-on-latency", TRUE, NULL);
  configure_source_for_ntp_sync (bin->src_elem);
  g_signal_connect (G_OBJECT (bin->src_elem), "pad-added",
      G_CALLBACK (cb_newpad3), bin);

  g_snprintf (elem_name, sizeof (elem_name), "tee_rtsp_elem%d", bin->bin_id);
  bin->tee_rtsp_pre_decode = gst_element_factory_make ("tee", elem_name);
  if (!bin->tee_rtsp_pre_decode) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  g_snprintf (elem_name, sizeof (elem_name), "dec_que%d", bin->bin_id);
  bin->dec_que = gst_element_factory_make ("queue", elem_name);
  if (!bin->dec_que) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  if (bin->rtsp_reconnect_interval_sec > 0) {
    printf("rtsp_reconnect_interval_sec: %u\n", bin->rtsp_reconnect_interval_sec);
    NVGSTDS_ELEM_ADD_PROBE (bin->rtspsrc_monitor_probe, bin->dec_que,
        "sink", rtspsrc_monitor_probe_func,
        GST_PAD_PROBE_TYPE_BUFFER,
        bin);
    install_mux_eosmonitor_probe = TRUE;
  }

  g_snprintf (elem_name, sizeof (elem_name), "decodebin_elem%d", bin->bin_id);
  bin->decodebin = gst_element_factory_make ("decodebin", elem_name);
  if (!bin->decodebin) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  g_signal_connect (G_OBJECT (bin->decodebin), "pad-added",
      G_CALLBACK (cb_newpad2), bin);
  g_signal_connect (G_OBJECT (bin->decodebin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);


  g_snprintf (elem_name, sizeof (elem_name), "src_que%d", bin->bin_id);
  bin->cap_filter = gst_element_factory_make ("queue", elem_name);
  if (!bin->cap_filter) {
    NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
    goto done;
  }

  g_mutex_init (&bin->bin_lock);

  g_snprintf(elem_name, sizeof(elem_name), "nvvidconv_elem%d", bin->bin_id);
  bin->nvvidconv = gst_element_factory_make("nvvideoconvert", elem_name);
  if (!bin->nvvidconv)
  {
    NVGSTDS_ERR_MSG_V("Could not create element 'nvvidconv_elem'");
    goto done;
  }
  caps = gst_caps_new_empty_simple("video/x-raw");
  feature = gst_caps_features_new("memory:NVMM", NULL);
  gst_caps_set_features(caps, 0, feature);

  bin->cap_filter1 =
      gst_element_factory_make("capsfilter", "src_cap_filter_nvvidconv");
  if (!bin->cap_filter1)
  {
    NVGSTDS_ERR_MSG_V("Could not create 'queue'");
    goto done;
  }

  g_object_set(G_OBJECT(bin->cap_filter1), "caps", caps, NULL);
  gst_caps_unref(caps);

  gst_bin_add_many (GST_BIN(bin->bin), 
                    bin->src_elem, 
                    bin->tee_rtsp_pre_decode,
                    bin->dec_que, 
                    bin->decodebin, 
                    bin->cap_filter, 
                    bin->nvvidconv, 
                    bin->cap_filter1, 
                    NULL);

  link_element_to_tee_src_pad(bin->tee_rtsp_pre_decode, bin->dec_que);
  NVGSTDS_LINK_ELEMENT (bin->dec_que, bin->decodebin);

  if (ctx)
    link_element_to_tee_src_pad(bin->tee_rtsp_pre_decode, ctx->recordbin);
    
  NVGSTDS_LINK_ELEMENT (bin->cap_filter, bin->nvvidconv);
  NVGSTDS_LINK_ELEMENT (bin->nvvidconv, bin->cap_filter1);
  NVGSTDS_BIN_ADD_GHOST_PAD (bin->bin, bin->cap_filter1, "src");

  ret = TRUE;

  g_timeout_add (1000, watch_source_status, bin);

done:
  if (!ret) {
    NVGSTDS_ERR_MSG_V ("%s failed", __func__);
  }
  return ret;

}

/**
 * Probe function to drop EOS events from nvstreammux when RTSP sources
 * are being used so that app does not quit from EOS in case of RTSP
 * connection errors and tries to reconnect.
 */
static GstPadProbeReturn
nvstreammux_eosmonitor_probe_func (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  if (info->type & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    GstEvent *event = (GstEvent *) info->data;
    if (GST_EVENT_TYPE (event) == GST_EVENT_EOS)
      return GST_PAD_PROBE_DROP;
  }
  return GST_PAD_PROBE_OK;
}

static gboolean
start_rtsp_streaming (guint updsink_port_num, guint64 udp_buffer_size,  string rtsp_port_string)
{
  GstRTSPMountPoints *mounts;
  GstRTSPMediaFactory *factory;
  char udpsrc_pipeline[512];
  char port_num_Str[64] = { 0 };
  string mapping_text;
  mapping_text.append("rtsp://127.0.0.1");
  //mapping_text.append(rtsp_port_string);
  mapping_text.append(std::getenv("RTSP_OUT_MAPPING"));
  if (udp_buffer_size == 0)
    udp_buffer_size = 512 * 1024;

  sprintf (udpsrc_pipeline,
      "( udpsrc name=pay0 port=%d buffer-size=%lu caps=\"application/x-rtp, media=video, "
      "clock-rate=90000, encoding-name=%s, payload=96 \" )",
      updsink_port_num, udp_buffer_size, "H264");

  sprintf (port_num_Str, "%d", stoi(rtsp_port_string));

  g_mutex_lock (&server_cnt_lock);

  server [server_count] = gst_rtsp_server_new ();
  g_object_set (server [server_count], "service", port_num_Str, NULL);

  mounts = gst_rtsp_server_get_mount_points (server [server_count]);

  factory = gst_rtsp_media_factory_new ();
  gst_rtsp_media_factory_set_launch (factory, udpsrc_pipeline);

  gst_rtsp_mount_points_add_factory (mounts, std::getenv("RTSP_OUT_MAPPING"), factory);

  g_object_unref (mounts);

  gst_rtsp_server_attach (server [server_count], NULL);

  server_count++;

  g_mutex_unlock (&server_cnt_lock);

  info_logger->info("RTSP IN : " + rtsp_in);
  info_logger->info("RTSP OUT launched at : " + mapping_text);
  cout << "\n *** DeepStream: Launched RTSP Streaming at: " << mapping_text << "***" <<endl;

  return TRUE;
}


static void
usage(const char *bin)
{
  g_printerr
    ("Usage: %s [-t infer-type]<elementary H264 file 1> ... <elementary H264 file n>\n",
      bin);
  g_printerr
    ("     -t infer-type: select form [infer, inferserver], infer by default\n");
}

#if (defined TRACKER)
  static gchar * get_absolute_file_path(gchar *cfg_file_path, gchar *file_path)
  {
    gchar abs_cfg_path[PATH_MAX + 1];
    gchar *abs_file_path;
    gchar *delim;

    if (file_path && file_path[0] == '/')
    {
      return file_path;
    }

    if (!realpath(cfg_file_path, abs_cfg_path))
    {
      g_free(file_path);
      return NULL;
    }

    /* Return absolute path of config file if file_path is NULL. */
    if (!file_path)
    {
      abs_file_path = g_strdup(abs_cfg_path);
      return abs_file_path;
    }

    delim = g_strrstr(abs_cfg_path, "/");
    *(delim + 1) = '\0';

    abs_file_path = g_strconcat(abs_cfg_path, file_path, NULL);
    g_free(file_path);

    return abs_file_path;
  }


  static gboolean set_tracker_properties(GstElement *nvtracker)
  {
    gboolean ret = FALSE;
    GError *error = NULL;
    gchar **keys = NULL;
    gchar **key = NULL;
    GKeyFile *key_file = g_key_file_new();

    if (!g_key_file_load_from_file(key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE,
                                  &error))
    {
      g_printerr("Failed to load config file: %s\n", error->message);
      return FALSE;
    }

    keys = g_key_file_get_keys(key_file, CONFIG_GROUP_TRACKER, NULL, &error);
    CHECK_ERROR(error);

    for (key = keys; *key; key++)
    {
      if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_WIDTH))
      {
        gint width =
            g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER,
                                  CONFIG_GROUP_TRACKER_WIDTH, &error);
        CHECK_ERROR(error);
        g_object_set(G_OBJECT(nvtracker), "tracker-width", width, NULL);
      }
      else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_HEIGHT))
      {
        gint height =
            g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER,
                                  CONFIG_GROUP_TRACKER_HEIGHT, &error);
        CHECK_ERROR(error);
        g_object_set(G_OBJECT(nvtracker), "tracker-height", height, NULL);
      }
      else if (!g_strcmp0(*key, CONFIG_GPU_ID))
      {
        guint gpu_id =
            g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER,
                                  CONFIG_GPU_ID, &error);
        CHECK_ERROR(error);
        g_object_set(G_OBJECT(nvtracker), "gpu_id", gpu_id, NULL);
      }
      else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE))
      {
        char *ll_config_file = get_absolute_file_path(TRACKER_CONFIG_FILE,
                                                      g_key_file_get_string(key_file,
                                                                            CONFIG_GROUP_TRACKER,
                                                                            CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
        CHECK_ERROR(error);
        g_object_set(G_OBJECT(nvtracker), "ll-config-file", ll_config_file, NULL);
      }
      else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE))
      {
        char *ll_lib_file = get_absolute_file_path(TRACKER_CONFIG_FILE,
                                                  g_key_file_get_string(key_file,
                                                                        CONFIG_GROUP_TRACKER,
                                                                        CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
        CHECK_ERROR(error);
        g_object_set(G_OBJECT(nvtracker), "ll-lib-file", ll_lib_file, NULL);
      }
      else if (!g_strcmp0(*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS))
      {
        gboolean enable_batch_process =
            g_key_file_get_integer(key_file, CONFIG_GROUP_TRACKER,
                                  CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
        CHECK_ERROR(error);
        g_object_set(G_OBJECT(nvtracker), "enable_batch_process",
                    enable_batch_process, NULL);
      }
      else
      {
        g_printerr("Unknown key '%s' for group [%s]", *key,
                  CONFIG_GROUP_TRACKER);
      }
    }

    ret = TRUE;
  done:
    if (error)
    {
      g_error_free(error);
    }
    if (keys)
    {
      g_strfreev(keys);
    }
    if (!ret)
    {
      g_printerr("%s failed", __func__);
    }
    return ret;
  }
#endif

int http_stream(int port, string link)
{
    MJPEGWriter test(port);
    bool ok;
    VideoCapture cap;
    Mat frame;
    while (true)
    {
        ok = cap.open(link);
        if (!ok)
        {
            printf("no cam found ;(.\n");
            usleep(60);
        }
        else
        {
                cap >> frame;
                if( !frame.empty() )
                {
                    test.write(frame);
                    frame.release();
                    test.start();
                    while(cap.isOpened())
                    {
                        cap >> frame; 
                        if( !frame.empty() ) 
                        {
                            time_t now = time(0);
                            tm *ltm = localtime(&now);
                            string jour,mois,heure,minute,seconde;
                            if ( ltm->tm_mday <10)
                                jour="0"+std::to_string(ltm->tm_mday);
                            else
                                jour=std::to_string(ltm->tm_mday);
                            if(1+ ltm->tm_mon <10)
                                mois= "0"+std::to_string(1+ ltm->tm_mon);
                            else
                                mois=std::to_string(1+ ltm->tm_mon);
                            if(ltm->tm_hour<10)
                                heure="0"+std::to_string(ltm->tm_hour);
                            else
                                heure=std::to_string(ltm->tm_hour);
                            if(ltm->tm_min<10)
                                minute="0"+std::to_string(ltm->tm_min);
                            else 
                                minute=std::to_string(ltm->tm_min);
                            if(ltm->tm_sec<10)
                                seconde="0"+std::to_string(ltm->tm_sec);
                            else
                                seconde=std::to_string(ltm->tm_sec);
                            string current_date=jour + "/"+mois +"/"+std::to_string(1900+ ltm->tm_year) + " "+heure+":"+minute+":"+seconde;
                            int xx = frame.cols/2;
                            putText(frame, //target image
                            current_date, //text
                            cv::Point(15, frame.rows-20), //top-left position
                            cv::FONT_HERSHEY_COMPLEX_SMALL,
                            3.0,
                            CV_RGB(255, 255, 255), //font color
                            2);
                            test.write(frame); 
                            frame.release();
                        }
                        else 
                        {
                            cap.release();
                            //cout << "no frame second" << endl;
                        } 
                    }
                    test.stop();
                }
                else
                {
                    //cout << "no frame first" << endl;
                    cap.release();
                }
        }
    }
}

int main (int argc, char *argv[])
{
  printf("app start..\n");
  appCtx[0] = (AppCtx *) g_malloc0 (sizeof (AppCtx));
  NvDsPipeline *pipeline = &appCtx[0]->pipeline;

  cout.rdbuf(NULL);

  info_logger->set_level(spdlog::level::info);
  info_logger->flush_on(spdlog::level::info);

  debug_logger->set_level(spdlog::level::debug);
  debug_logger->flush_on(spdlog::level::debug);

  time_t t = time(NULL);
  struct tm lt = {0};
  localtime_r(&t, &lt);
  time_zone = lt.tm_zone;
  info_logger->info("Time zone : " + time_zone);
  offset_tz = lt.tm_gmtoff / 3600;
  info_logger->info("Time offset : " + to_string(offset_tz));
  
  #if (defined TRACKER && defined ANALYTICS)
  GstElement *streammux = NULL, 
              *sink = NULL, 
              *pgie = NULL, 
              *sgie1 = NULL,
              *nvosd = NULL,
              *queue1 = NULL,
              *queue2 = NULL, 
              *nvvidconv2 = NULL, 
              *caps_filter = NULL, 
              *encoder = NULL, 
              *codecparser = NULL, 
              *rtppay = NULL, 
              *tiler = NULL,
              *nvtracker = NULL,
              *nvdsanalytics=NULL;
  #else
  #if (defined TRACKER)
  GstElement *streammux = NULL, 
              *sink = NULL, 
              *pgie = NULL, 
              *sgie1 = NULL,
              *nvtracker = NULL,
              *nvosd = NULL, 
              *queue1 = NULL,
              *queue2 = NULL, 
              *nvvidconv2 = NULL, 
              *caps_filter = NULL, 
              *encoder = NULL, 
              *codecparser = NULL, 
              *rtppay = NULL, 
              *tiler = NULL;
  #else
  #if (defined ANALYTICS)
  GstElement *streammux = NULL, 
              *sink = NULL, 
              *pgie = NULL, 
              *sgie1 = NULL,
              *nvosd = NULL,
              *queue1 = NULL, 
              *queue2 = NULL, 
              *nvvidconv2 = NULL, 
              *caps_filter = NULL, 
              *encoder = NULL, 
              *codecparser = NULL, 
              *rtppay = NULL, 
              *tiler = NULL,
              *nvdsanalytics = NULL;
  #else
  GstElement *streammux = NULL, 
              *sink = NULL, 
              *pgie = NULL, 
              *sgie1 = NULL,
              *nvosd = NULL, 
              *queue1 = NULL,
              *queue2 = NULL, 
              *nvvidconv2 = NULL, 
              *caps_filter = NULL, 
              *encoder = NULL, 
              *codecparser = NULL, 
              *rtppay = NULL, 
              *tiler = NULL;
  #endif
  #endif
  #endif
  GstCaps *caps = NULL;
  GstBus *bus = NULL;
  guint i=0;
  guint tiler_rows, tiler_columns;
  GstPad *osd_sink_pad = NULL;
  guint pgie_batch_size;
  const char *infer_plugin = NVINFER_PLUGIN;
  gboolean ret = FALSE;

  /* define numsorces from argc */
  if (argc < 2) {
    g_printerr ("Usage: %s <uri>\n", argv[0]);
    return -1;
  }
  num_sources = argc - 1;
  /* define numsorces from argc */
  rtsp_in = std::getenv("RTSP_STREAM");


  // cout << "-------------- RTSP IN" << endl;
  // // cout << "argv[0]: " << argv[0] << endl;
  // // cout << "argv[1]: " << argv[1] << endl;
  // // cout << "argv[2]: " << argv[2] << endl;
  // cout << "rtsp_in: " << rtsp_in << endl;
  // cout << "----------------------" << endl;

  nvds_version(&nvds_lib_major_version, &nvds_lib_minor_version);

  gst_init (&argc, &argv);

  pipeline->pipeline = gst_pipeline_new ("pipeline");
  if (!pipeline->pipeline) {
    NVGSTDS_ERR_MSG_V ("Failed to create pipeline");
  }

  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline->pipeline));
  pipeline->bus_id = gst_bus_add_watch (bus, bus_callback, appCtx[0]);
  gst_object_unref (bus);

  pipeline->multi_src_bin.reset_thread = NULL;

  pipeline->multi_src_bin.bin = gst_bin_new ("multi_src_bin");
  if (!pipeline->multi_src_bin.bin) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'multi_src_bin'");
    goto done;
  }
  g_object_set (pipeline->multi_src_bin.bin, "message-forward", TRUE, NULL);

  pipeline->multi_src_bin.streammux = gst_element_factory_make ("nvstreammux", "src_bin_muxer");
  if (!pipeline->multi_src_bin.streammux) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'src_bin_muxer'");
    goto done;
  }
  gst_bin_add (GST_BIN (pipeline->multi_src_bin.bin), pipeline->multi_src_bin.streammux);
  
  /* create rtsp src bin */
  for (i = 0; i < num_sources; i++) {
    printf("%s: %u\n", "start createing source bins", i);
    GstPad *sinkpad, *srcpad;
    gchar elem_name[50];
    g_snprintf (elem_name, sizeof (elem_name), "src_sub_bin%d", i);
    pipeline->multi_src_bin.sub_bins[i].bin = gst_bin_new (elem_name);
    if (!pipeline->multi_src_bin.sub_bins[i].bin) {
      NVGSTDS_ERR_MSG_V ("Failed to create '%s'", elem_name);
      goto done;
    }

    pipeline->multi_src_bin.sub_bins[i].bin_id = pipeline->multi_src_bin.sub_bins[i].source_id = i;
    pipeline->multi_src_bin.live_source = TRUE;
    pipeline->multi_src_bin.sub_bins[i].eos_done = TRUE;
    pipeline->multi_src_bin.sub_bins[i].reset_done = TRUE;
  
    printf("argv[2]: %s\n", argv[i+1]);
    if(!create_rtsp_src_bin (i, argv[i + 1], &pipeline->multi_src_bin.sub_bins[i])){
      g_printerr ("Failed to create source bin. Exiting.\n");
    }
    gst_bin_add (GST_BIN (pipeline->multi_src_bin.bin), pipeline->multi_src_bin.sub_bins[i].bin);

    if (!link_element_to_streammux_sink_pad (pipeline->multi_src_bin.streammux,
            pipeline->multi_src_bin.sub_bins[i].bin, i)) {
      NVGSTDS_ERR_MSG_V ("source %d cannot be linked to mux's sink pad %p\n", i, pipeline->multi_src_bin.streammux);
      goto done;
    }
    pipeline->multi_src_bin.num_bins++;
    printf("pipeline->multi_src_bin.num_bins: %d\n", pipeline->multi_src_bin.num_bins);
  }
  /* create rtsp src bin */

  NVGSTDS_BIN_ADD_GHOST_PAD (pipeline->multi_src_bin.bin, pipeline->multi_src_bin.streammux, "src");

  if (install_mux_eosmonitor_probe) {
    NVGSTDS_ELEM_ADD_PROBE (pipeline->multi_src_bin.nvstreammux_eosmonitor_probe, pipeline->multi_src_bin.streammux,
        "src", nvstreammux_eosmonitor_probe_func,
        GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
        &pipeline->multi_src_bin);
  }

done:

  loop = g_main_loop_new (NULL, FALSE);

  pgie = gst_element_factory_make (infer_plugin, "primary-nvinference-engine");  
  sgie1 = gst_element_factory_make (infer_plugin, "secondary-nvinference-engine"); 
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  #if (defined TRACKER)
    nvtracker = gst_element_factory_make("nvtracker", "nvtracker");
  #endif
  #if (defined ANALYTICS)
    nvdsanalytics = gst_element_factory_make ("nvdsanalytics", "nvdsanalytics");
  #endif
  g_object_set (G_OBJECT (nvosd), "display-text", 1, NULL);
  g_object_set (G_OBJECT (nvosd), "display-clock", 1, NULL);
  g_object_set (G_OBJECT (nvosd), "display-bbox", 1, NULL);
  g_object_set (G_OBJECT (nvosd), "display-mask", 1, NULL);
  g_object_set (G_OBJECT (nvosd), "process-mode", 2, NULL);
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");
  nvvidconv2 = gst_element_factory_make ("nvvideoconvert", "convertor2");
  guint gpu_id = 0;
  caps_filter = gst_element_factory_make ("capsfilter", "capsfilter");
  caps = gst_caps_from_string ("video/x-raw(memory:NVMM), format=I420");
  encoder = gst_element_factory_make ("nvv4l2h264enc", "encoder");
  guint profile = 0;
  guint bitrate = 1000000;
  guint iframeinterval = 60;
  codecparser = gst_element_factory_make ("h264parse", "h264-parser2");
  rtppay = gst_element_factory_make ("rtph264pay", "rtppay");
  sink = gst_element_factory_make ("udpsink", "udpsink");

  g_object_set (G_OBJECT (pipeline->multi_src_bin.streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batch-size", num_sources,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }
  g_object_set (G_OBJECT (pgie), "batch-size", 1, NULL);
  g_object_set (G_OBJECT (pgie), "config-file-path", INFER_PGIE_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT (sgie1), "config-file-path", INFER_SGIE1_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT (nvvidconv2), "gpu-id", gpu_id, NULL);
  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);
  g_object_set (G_OBJECT (encoder), "profile", profile, NULL);
  g_object_set (G_OBJECT (encoder), "iframeinterval", iframeinterval, NULL);
  g_object_set (G_OBJECT (encoder), "bitrate", bitrate, NULL);
  g_object_set (G_OBJECT (encoder), "preset-level", 1, NULL);
  g_object_set (G_OBJECT (encoder), "insert-sps-pps", 1, NULL);
  g_object_set (G_OBJECT (encoder), "bufapi-version", 1, NULL);

  int RTSP_UDP_PORT = std::stoi(std::getenv("RTSP_OUT_UDP_PORT"));
  g_object_set (G_OBJECT (sink), "host", "224.224.255.255", "port",
      RTSP_UDP_PORT, "async", FALSE, "sync", 0, NULL);

  if (!pgie) {
    g_printerr ("pgie could not be created. Exiting.\n");
    return -1;
  }
  if (!sgie1) {
    g_printerr ("sgie1 could not be created. Exiting.\n");
    return -1;
  }
  if (!tiler) {
    g_printerr ("tiler could not be created. Exiting.\n");
    return -1;
  }
  if (!nvosd) {
    g_printerr ("nvosd could not be created. Exiting.\n");
    return -1;
  }
  if (!sink) {
    g_printerr ("sink could not be created. Exiting.\n");
    return -1;
  }
  if (!queue2 || !nvvidconv2 || !caps_filter || !encoder || !codecparser ) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  if (!rtppay) {
    g_printerr ("rtppay could not be created. Exiting.\n");
    return -1;
  }
  #if (defined TRACKER)
    if (!set_tracker_properties(nvtracker))
    {
      g_printerr("Failed to set tracker properties. Exiting.\n");
      return -1;
    }
  #endif
  #if (defined ANALYTICS)
    g_object_set (G_OBJECT (nvdsanalytics), "config-file", NVDSANALYTICS_CONFIG_FILE, NULL);
  #endif
  /* a tee after the tiler which shall be connected to sink(s) */
  pipeline->tiler_tee = gst_element_factory_make ("tee", "tiler_tee");
  if (!pipeline->tiler_tee) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'tiler_tee'");
  }
  /* Create demuxer only if tiled display is disabled. */
  pipeline->demuxer =
      gst_element_factory_make ("nvstreamdemux", "demuxer");
  if (!pipeline->demuxer) {
    NVGSTDS_ERR_MSG_V ("Failed to create element 'demuxer'");
  }

  /* setting tiler */ 
  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);
  #if (defined TRACKER && defined ANALYTICS)
    gst_bin_add_many (GST_BIN (pipeline->pipeline),
        pipeline->multi_src_bin.bin, 
        pipeline->multi_src_bin.streammux,
        queue1,
        pgie,
        nvtracker,
        sgie1,
        queue2,
        nvdsanalytics,
        tiler, 
        nvosd, 
        nvvidconv2, 
        caps_filter, 
        encoder, 
        codecparser, 
        rtppay, 
        sink, 
        NULL);
    if (!gst_element_link_many (pipeline->multi_src_bin.bin,
                                queue1, 
                                pgie,
                                nvtracker,
                                sgie1,
                                queue2,   
                                nvdsanalytics,
                                tiler, 
                                nvosd, 
                                queue2, 
                                nvvidconv2, 
                                caps_filter, 
                                encoder, 
                                codecparser, 
                                rtppay, 
                                sink, 
                                NULL)) 
    {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }
  #else
  #if (defined TRACKER)
   cout << "here" << endl;
    gst_bin_add_many (GST_BIN (pipeline->pipeline),
        pipeline->multi_src_bin.bin, 
        pipeline->multi_src_bin.streammux,
        queue1,
        pgie,
        nvtracker,
        sgie1,
        queue2,
        tiler,
        nvosd, 
        nvvidconv2, 
        caps_filter, 
        encoder, 
        codecparser, 
        rtppay, 
        sink, 
        NULL);
    if (!gst_element_link_many (pipeline->multi_src_bin.bin, 
                                queue1,
                                pgie, 
                                nvtracker, 
                                sgie1,
                                queue2, 
                                tiler,
                                nvosd,
                                nvvidconv2, 
                                caps_filter, 
                                encoder, 
                                codecparser, 
                                rtppay, 
                                sink, 
                                NULL)) 
    {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }
  #else
  #if (defined ANALYTICS)
    gst_bin_add_many (GST_BIN (pipeline->pipeline),
          pipeline->multi_src_bin.bin, 
          pipeline->multi_src_bin.streammux,
          queue1,
          pgie, 
          sgie1,
          queue2,
          nvdsanalytics, 
          tiler, 
          nvosd, 
          nvvidconv2, 
          caps_filter, 
          encoder, 
          codecparser, 
          rtppay, 
          sink, 
          NULL);
    if (!gst_element_link_many (pipeline->multi_src_bin.bin,
                                queue1, 
                                pgie,
                                sgie1,
                                queue2,
                                nvdsanalytics,
                                tiler, 
                                nvosd, 
                                queue2, 
                                nvvidconv2, 
                                caps_filter, 
                                encoder, 
                                codecparser, 
                                rtppay, 
                                sink, 
                                NULL)) 
    {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }
  #else
    gst_bin_add_many (GST_BIN (pipeline->pipeline),
        pipeline->multi_src_bin.bin, 
        pipeline->multi_src_bin.streammux,
        queue1,
        pgie,
        sgie1,
        queue2, 
        tiler, 
        nvosd, 
        nvvidconv2, 
        caps_filter, 
        encoder, 
        codecparser, 
        rtppay, 
        sink, 
        NULL);
    if (!gst_element_link_many (pipeline->multi_src_bin.bin, 
                                queue1,
                                pgie,
                                sgie1,
                                queue2, 
                                tiler, 
                                nvosd,
                                nvvidconv2, 
                                caps_filter, 
                                encoder, 
                                codecparser, 
                                rtppay, 
                                sink, 
                                NULL)) 
    {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }
  #endif
  #endif
  #endif
  // #if (defined TRACKER)
  //   osd_sink_pad = gst_element_get_static_pad (nvtracker, "src");
  // #else
  //   osd_sink_pad = gst_element_get_static_pad (pgie, "src");
  // #endif
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);

  ret = TRUE;
  int HTTP_RTSP_IN_PORT = 15;
  int HTTP_RTSP_OUT_PORT = 15 ;
  int RTSP_PORT = 15; // par defaut
  if (getenv("HTTP_IN_PORT") != NULL)
    HTTP_RTSP_IN_PORT = std::stoi(std::getenv("HTTP_IN_PORT"));
  if (getenv("HTTP_OUT_PORT") != NULL)
  {
    cout << " I'am not null i'am " <<  getenv("HTTP_OUT_PORT") << endl; 
    HTTP_RTSP_OUT_PORT = std::stoi(std::getenv("HTTP_OUT_PORT"));
  }
  
  if (getenv("RTSP_OUT_PORT") != NULL)
    RTSP_PORT = std::stoi(std::getenv("RTSP_OUT_PORT"));
  string mapping_text;
  mapping_text.append("rtsp://127.0.0.1");
  mapping_text.append(std::getenv("RTSP_OUT_PORT"));
  mapping_text.append(std::getenv("RTSP_OUT_MAPPING"));
  if (RTSP_PORT != 15)
  {
    cout << "RTSP Enable" << endl;
    ret = start_rtsp_streaming (RTSP_UDP_PORT, 0,std::getenv("RTSP_OUT_PORT"));
  }
  if (HTTP_RTSP_OUT_PORT != 15)
  {
    //thread t1(http_stream,HTTP_RTSP_OUT_PORT, mapping_text);
    cout << "HTTP OUT Enable at port : " << HTTP_RTSP_OUT_PORT << endl;
    std::future<int> f1 =std::async(std::launch::async,http_stream,HTTP_RTSP_OUT_PORT, mapping_text);
  }
  if (HTTP_RTSP_IN_PORT != 15)
  {
    cout << "HTTP IN Enable" << endl;
    string env_p = std::getenv("RTSP_STREAM");
    thread t2(http_stream,HTTP_RTSP_IN_PORT, env_p);
    //std::future<int> f2 =std::async(std::launch::async,http_stream,HTTP_RTSP_IN_PORT, env_p);
  }
  if (ret != TRUE) {
    g_print ("%s: start_rtsp_straming function failed\n", __func__);
  }

  

  //Start recording to video with ffmpeg
  // start_video = time(0);
  // recording_file_name.append(to_string(start_video));
  // recording_file_name.append("_record.ts");

  string command_dir1;
  command_dir1.append("mkdir -p /phoenix/media/alerts");
  system(command_dir1.c_str());

  command_dir1 = "";
  command_dir1.append("mkdir -p /phoenix/media/requested");
  system(command_dir1.c_str());


  /*string command;
  command.append("ffmpeg -hide_banner -loglevel panic -fflags nobuffer -flags low_delay -strict experimental -rtsp_transport tcp -i '");
  // command.append(mapping_text);
  command.append(rtsp_in);
  //command.append(" -r ");
  //command.append(to_string(30.0/DROP_FRAME_INTERVAL));
  // command.append(" -filter:v fps=30 ");
  command.append("' -vcodec copy -an ");
  command.append("-f segment -segment_list out.list -segment_time " + to_string(video_segment_duration) + " -segment_atclocktime 1 -strftime 1 \"/phoenix/media/%Y-%m-%d_%H-%M-%S.ts\"");
  //command.append(recording_file_name);
  command.append(" &");
  
  cout << command << endl;
  info_logger->info("RTSP IN : " + rtsp_in);
  system(command.c_str());*/
    

  getParams();
  sleep(video_duration); // need at least that amount of video to start any treatment

  g_print ("Now playing...\n");
  gst_element_set_state (pipeline->pipeline, GST_STATE_PLAYING);
  g_print ("Running...\n");
  g_main_loop_run (loop);
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline->pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline->pipeline));
  g_source_remove (pipeline->bus_id);
  g_main_loop_unref (loop);

  string command="";
  command.append("pkill -f ffmpeg"); //stop recording
  system(command.c_str());

  return 0;
}


