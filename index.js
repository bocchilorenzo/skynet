/**
 * INSTRUCTIONS
 *  - create a folder in the root of the project called "current_dataset"
 *  and put both the input files, one for testing and one for training(*)
 *  - fill the .env file with the wanted values
 * 
 * (*) the values need to be separated by commas. It is recommended to put the expected output in the input file
 *     and using the same column name for the target (OUTPUT_COLUMN_NAME) to get a better output
 * 
 * WHAT DO THE VALUES IN .ENV MEAN?
 *  - OUTPUT_CSV_NAME = file name for the output csv
 *  - INPUT_NAME = file name for the input csv to test, not for training
 *  - EPOCH = number of cycles for the training
 *  - LEARNING_RATE = learning rate
 *	- HIDDEN_NEURONS = number of hidden nodes
 *  - PREV_TRAINING = true if training was made and synapses data was saved, false otherwise
 *  - NETWORK_NAME = name of the network, used to save training data to be able to reuse the network with different data
 *  - TRAINING_NAME = name of the training file
 *  - OUTPUT_COLUMN_NAME = name of the target column in the csv
 *  - INPUT_FIELDS = name of the input columns
 */

//IMPORT LIBRARIES
console.log("Loading network and libraries...")
import { NeuralNetwork } from './nn' //import the neural network
const math = require('mathjs') // math library
const parse = require('csv-parse/lib/sync'); //csv utilities
const fs = require("fs");
const generate = require('csv-generate')
require("dotenv").config();

//IMPORT DATA
console.log("Loading file...")
let outName = process.env.OUTPUT_CSV_NAME
let inputName = process.env.INPUT_NAME
const input = fs.readFileSync("./current_dataset/" + inputName + ".csv").toString(); //read the input file
const records = parse(input, {
    columns: true,
    skip_empty_lines: true
});

//DEFINE INPUT FIELDS
console.log("Loading data...")
let inputFields = process.env.INPUT_FIELDS.split(",")
let hiddenNodes = parseInt(process.env.HIDDEN_NEURONS)

//CREATE VARIABLES
let target = math.matrix() //create an empty matrix to use for target in training
let matInput = math.matrix() //create an empty matrix to use as input
let matTrain = math.matrix() //create an empty matrix to use for training
let mat //temporary matrix to use in the coming cycle
let epoch = parseInt(process.env.EPOCH)
let lr = parseFloat(process.env.LEARNING_RATE)
let output = 1 //number of outputs
let previousTraining = math.boolean(process.env.PREV_TRAINING)
let netName = process.env.NETWORK_NAME

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
    let trainName = process.env.TRAINING_NAME
    const train = fs.readFileSync("./current_dataset/" + trainName + ".csv").toString(); //read the training file
    const recordsTrain = parse(train, {
        columns: true,
        skip_empty_lines: true
    });

    //CREATE TARGET ARRAY
    let outputName = process.env.OUTPUT_COLUMN_NAME
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
fs.mkdir('./results/' + netName + "/", { recursive: true }, (err) => {
    if (err) throw err;
})
const writeStream = fs.createWriteStream('./results/' + netName + '/' + outName + ".csv");
let targetArr = []
let outputName = process.env.OUTPUT_COLUMN_NAME
try {
    targetArr = records.map(rec => rec[outputName]);
    //writeStream.write("sep=," + '\n'); //enable to have a better reading experience in Excel
    writeStream.write("Expected,Received,isCorrect" + '\n');
    for (let i = 0; i < result._size[0]; i++) {
        writeStream.write(targetArr[i] + "," + math.flatten(result)._data[i]);
        if (parseFloat(targetArr[i]) == 1) {
            parseFloat(targetArr[i]) - math.flatten(result)._data[i] < 0.5 ? writeStream.write(",1" + '\n') : writeStream.write(",0" + '\n')
        }
        else {
            parseFloat(targetArr[i]) + math.flatten(result)._data[i] < 0.5 ? writeStream.write(",1" + '\n') : writeStream.write(",0" + '\n')
        }
    }
} catch (err) {
    console.log(err)
    for (let i = 0; i < result._size[0]; i++) {
        writeStream.write(math.flatten(result)._data[i] + '\n');
    }
}
console.log("Testing completed! Check the results in the 'result' directory")

/*
//DATASET ANALYSIS
const ds = fs.readFileSync("./current_dataset/dataset.csv").toString(); //read the input file
const recordsDs = parse(ds, {
    columns: true,
    skip_empty_lines: true
});
let values = {}
let fields = inputFields
fields.push(process.env.OUTPUT_COLUMN_NAME)
//exclude last value as it's the output
for (let i = 1; i < fields.length - 1; i++) {
    values[fields[i]] = 0
}
let desiredResult = "" //set the result to monitor
for (let i = 0; i < recordsDs.length; i++) {
    mat = math.matrix([[recordsDs[i][fields[0]]]]);
    if (fields.length > 1) {
        for (let x = 1; x < fields.length; x++) {
            if (fields[x] == undefined) {
                mat = math.concat(mat, math.matrix([[0]]));
            }
            else {
                mat = math.concat(mat, math.matrix([[recordsDs[i][fields[x]]]]));
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
for (let i = 0; i < matInput._size[0]; i++) {
    //if the result for that line is that count the values
    if (matInput._data[i][15] == desiredResult) {
        for (let x = 0; x < matInput._data[i].length - 1; x++) {
            values[fields[x]] += 1
        }
    }
}
console.log(values)
*/