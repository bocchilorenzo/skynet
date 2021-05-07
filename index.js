//IMPORT LIBRARIES
console.log("Loading network and libraries...")
import { NeuralNetwork } from './nn' //import the neural network
const math = require('mathjs') // math library
const parse = require('csv-parse/lib/sync'); //csv utilities
const fs = require("fs");
const generate = require('csv-generate')

//CSV FILES NEED TO BE SEPARATED BY COMMAS
//for the input files, both the training csv and the test csv, create a folder named "current_dataset" and put the files there

//IMPORT DATA
console.log("Loading file...")
let fileName = "" //file name for the output csv
let inputName = "" //file name for the input csv to test, not for training
const input = fs.readFileSync("./current_dataset/" + inputName + ".csv").toString(); //read the input file
const records = parse(input, {
    columns: true,
    skip_empty_lines: true
});

//DEFINE INPUT FIELDS
console.log("Loading data...")
let inputFields = []
let hiddenNodes = inputFields.length //set the number of hidden nodes

//CREATE VARIABLES
let target = math.matrix() //create an empty matrix to use for target in training
let matInput = math.matrix() //create an empty matrix to use as input
let matTrain = math.matrix() //create an empty matrix to use for training
let mat //temporary matrix to use in the coming cycle
let epoch = 5000 //number of cycles for the training
let lr = .1 //learning rate
let output = 1 //number of outputs
let previousTraining = false //true if training was made and synapses data was saved, false otherwise
let netName = "" //name of the network, used to save training data to be able to reuse the network with different data

//CREATE THE INPUT MATRIX
for (let i = 0; i < records.length; i++) {
    mat = math.matrix([[parseFloat(records[i][inputFields[0]])]]);
    if (inputFields.length > 1) {
        for (let x = 1; x < inputFields.length; x++) {
            if (inputFields[x] == undefined) {
                mat = math.concat(mat, math.matrix([[0]]));
            }
            else {
                mat = math.concat(mat, math.matrix([[parseFloat(records[i][inputFields[x]])]]));
            }
        }
    }
    if (i == 0) {
        matInput = mat
    }
    else {
        matInput = math.concat(matInput, mat, 0)
    }
}

//CREATE NETWORK
const nn = new NeuralNetwork(inputFields.length, hiddenNodes, output, epoch, lr, netName);

//LOAD OR CREATE TRAINING DATA
if (previousTraining) {
    console.log("Loading training data...")
    const synapses = require("./connections/" + netName); //require previous training data
    nn.loadSynapses(synapses.syn0, synapses.syn1)
}
else {
    let trainName = ""
    const train = fs.readFileSync("./current_dataset/" + trainName + ".csv").toString(); //read the training file
    const recordsTrain = parse(train, {
        columns: true,
        skip_empty_lines: true
    });

    //CREATE TARGET ARRAY
    let outputName = "" //name of the output column in the csv
    let targetArr = recordsTrain.map(rec => rec[outputName]) //get the expected target values

    //CREATE THE TRAINING AND TARGET MATRIX
    for (let i = 0; i < recordsTrain.length; i++) {
        mat = math.matrix([[parseFloat(recordsTrain[i][inputFields[0]])]]);
        if (inputFields.length > 1) {
            for (let x = 1; x < inputFields.length; x++) {
                if (inputFields[x] == undefined) {
                    mat = math.concat(mat, math.matrix([[0]]));
                }
                else {
                    mat = math.concat(mat, math.matrix([[parseFloat(recordsTrain[i][inputFields[x]])]]));
                }
            }
        }
        if (i == 0) {
            matTrain = mat
            target = math.matrix([[parseFloat(targetArr[i])]])
        }
        else {
            matTrain = math.concat(matTrain, mat, 0)
            target = math.concat(target, math.matrix([[parseFloat(targetArr[i])]]), 0)
        }
    }
    console.log("Training started...")
    nn.train(matTrain, target); //train the network
}
console.log("Testing started...")
let result = nn.predict(matInput); //test the network

//SAVE RESULTS
fs.mkdir('./results/', { recursive: true }, (err) => {
    if (err) throw err;
});
const writeStream = fs.createWriteStream('./results/' + fileName + "_output.csv");
for (let i = 0; i < result._size[0]; i++) {
    writeStream.write(math.flatten(result)._data[i] + '\n');
}
console.log("Testing completed! Check the results in the 'result' directory")