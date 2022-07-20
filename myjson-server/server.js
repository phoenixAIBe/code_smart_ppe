const jsonServer = require("json-server");
const server = jsonServer.create();
const router = jsonServer.router("../parameters.json");
const middlewares = jsonServer.defaults();
const fs = require("fs");
// const execSync = require('child_process').execSync;
const spawn = require('child_process').spawn;




server.use(jsonServer.bodyParser);
server.use(middlewares);

function makeid(length) {
  var result           = '';
  var characters       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  var charactersLength = characters.length;
  for ( var i = 0; i < length; i++ ) {
    result += characters.charAt(Math.floor(Math.random() * 
charactersLength));
 }
 return result;
}

function toTimestamp(year,month,day,hour,minute,second){
  var datum = new Date(year,month-1,day,hour,minute,second);
  return datum.getTime()/1000;
}

function createVideo(id, req){
  const files_list = fs.readdirSync('../records');
  // console.log(files_list);
  const files = {};
  
  files_list.forEach(element => 
    {
      if(element.includes(".ts"))
      {
        let splitted = element.split("_");
        let [year, month, day] = splitted[0].split("-");
        let [hours, minutes, seconds] = splitted[1].replace(".ts", "").split("-");
        files[element] = toTimestamp(year, month, day, hours, minutes, seconds);
      }
    }); 
      
  //Sort dictionary
  // Create items array
  var items = Object.keys(files).map(function(key) {
    return [key, files[key]];
  });

  // Sort the array based on the second element
  items.sort(function(first, second) {
    return first[1] - second[1];
  });

  console.log(items);

  let list_of_files = [];
  let first_found = false;
  items.forEach((element, index) => {
    if(((!first_found && req.body.start >= element[1]) && (items[index+1][1] > req.body.start)) || (first_found && element[1] < req.body.end))
    {
      list_of_files.push(element[0]);
      first_found = true;
    }
  });

  console.log(list_of_files);

  const file = fs.createWriteStream('list.txt');
  //file.on('error', function(err) { /* error handling */ });
  list_of_files.forEach(function(v) { file.write("file ../records/" + v + "\n"); });
  file.end();

  // const command = "ffmpeg -f concat -safe 0 -i list.txt -c copy /var/www/html/videos/" + id + ".mp4"; 
  // console.log(command);
  // // import { execSync } from 'child_process';  // replace ^ if using ES modules
  // const output = execSync(command, { encoding: 'utf-8' });  // the default is 'buffer'


  const cmd = 'ffmpeg';

  const args = [
      '-f', 'concat',
      '-safe', '0',
      '-i', 'list.txt', 
      '-c', 'copy', 
      '/var/www/html/videos/' + id + '.mp4'
  ];

  const proc = spawn(cmd, args);

  // proc.stdout.on('data', function(data) {
  //     console.log(data);
  // });

  // proc.stderr.setEncoding("utf8")
  // proc.stderr.on('data', function(data) {
  //     console.log(data);
  // });

  // proc.on('close', function() {
  //     console.log('finished');
  // });

}


// Custom middleware to access POST methods.
// Can be customized for other HTTP method as well.
server.use((req, res, next) => {
  console.log("POST request listener");
  const body = req.body;
  console.log(body);
  if (req.method === "POST") {
    // If the method is a POST echo back the name from request body
    const start = body.start;
    const end = body.end;
    const first_available_timestamp = 0;
    const now = Date.now() / 1000;
    console.log("Actual timestamp: " + now);
    if(start >= first_available_timestamp && end <= now)
    {
      //db.get(name).insert(req.body).value();
      const id = makeid(32);
      createVideo(id, req);
      const link = "http://10.0.0.65/videos/" + id + ".mp4";
      req.body.link = link;
      req.body.status = "ok";
      next(); //add info into db
    }
    else if(end > now && start < first_available_timestamp){
      req.body.status = "Start before first available timestamp and end after now"
      res.status(404).json(req.body);
    }
    else if(end > now)
    {
      req.body.status = "End after now";
      res.status(404).json(req.body);
    }
    else if(start < first_available_timestamp)
    {
      req.body.status = "Start before first available timestamp";
      res.status(404).json(req.body);
    }
    else
    {
      res.status(500);
    }
    
  }else{
      //Not a post request. Let db.json handle it
      next();
  }  
});

server.use(router);

server.listen(3500, () => {
  console.log("JSON Server is running");
});