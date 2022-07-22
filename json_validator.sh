#!/bin/sh
myjson=init
launcher=true
#touch /opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/smart_waste/json-detection/json_to_stream.json

FILE1=/opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/smart_ppe/json-detection/json_to_stream.json
FILE2=/opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/smart_ppe/json-detection/detection_results.json

(nohup /root/.nvm/versions/node/v14.0.0/bin/json-server --watch $FILE1 --port 2000 --host 0.0.0.0 > json-server.log &)

while true; do

    if [ -f "$FILE2" ]; then
        if [ -f "$FILE1" ]; then
            #echo "Checking if detection_results.json is different"
            if [ "$myjson" != "{$(cat json-detection/detection_results.json) ]}" ]; then
                echo "Updating json file"
                myjson="{$(cat json-detection/detection_results.json) ]}"

                json validate --schema-file=json-detection/schema.json --document-json="{$(cat json-detection/detection_results.json) ]}"
                if [ $? -eq 0 ]; then
                    echo "Parsed JSON successfully"
                    echo $myjson > $FILE1
                    (nohup /root/.nvm/versions/node/v14.0.0/bin/json-server --watch $FILE1 --port 2000 --host 0.0.0.0 > json-server.log &)
                else
                    echo "Failed to parse JSON"
                fi
            fi
        fi
    fi

    currenttime=$(date +%H:%M:%S)
    if [[ "$currenttime" > "23:59:57" ]] && [[ "$currenttime" < "00:00:03" ]]; then
        (nohup /root/.nvm/versions/node/v14.0.0/bin/json-server --watch $FILE1 --port 2000 --host 0.0.0.0 > json-server.log &)
    fi

    sleep 5

done

