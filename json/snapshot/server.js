const express = require('express');
const {spawn} = require('child_process');
const app = express();
const port = 5000;
// Define the static file path
app.use(express.static(__dirname+'/public'));
app.get('/snapshot', function (req, res) 
{
    const python = spawn('python3', ['get_snapshot.py']);
    python.stdout.on('data', function (data) {
        console.log('Pipe data from python script ...');
        dataToSend = data.toString();});
    res.sendFile(__dirname + '/index.html');
})
app.listen(port, () => console.log('The server running on Port '+port));