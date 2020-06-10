const tf = require('@tensorflow/tfjs');
const fs = require('fs');

const stemmer = require('./stemmer_module');
const tokenize = require('./tokenize_module');

let RAW_DATA;
let model_responses = [];

function getRawData(dataUrl) {
    RAW_DATA = fs.readFileSync(dataUrl);
    const rawData = JSON.parse(RAW_DATA);
    return rawData;
}

//Load responses
function loadResponses(responses) {
    for (let i in responses) {
        model_responses.push(responses[i]);
    }
    return model_responses;
}

//Get respond
function getRespond(className) {
    let allResponses = model_responses;
    let botResponse;
    for (let i in allResponses) {
        const tempResponse = allResponses[i];
        if (tempResponse[0] == className) {
            botResponse = tempResponse[1];
            const responseIndex = Math.floor(Math.random() * botResponse.length);
            botResponse = botResponse[responseIndex];
        }
    }
    return botResponse;
}

//Refine sentences
function refineSentence(sentence) {
    //tokenize the pattern
    let stemmed_words = [];
    let sentence_words = tokenize(sentence.toLowerCase());
    //stem each word
    for (let i in sentence_words) {
        stemmed_words.push(stemmer(sentence_words[i].token));
    };
    //console.log('stems: ', stemmed_words);
    return stemmed_words;
}

//Create Bag of Words Tensor
function createWordTensor(sentence, words_doc, show_details = false) {
    //tokenize the pattern
    let sentence_words = refineSentence(sentence);

    //bag of words
    let bag = tf.zeros([1, words_doc.length]).dataSync();

    for (let s in sentence_words) {
        for (let i of words_doc.entries()) {
            if (i[1] == sentence_words[s]) {
                bag[i[0]] = 1;
                if (show_details) {
                    console.log("Found in bag: " + i[1]);
                }
            }
        }
    }
    return tf.tensor1d(bag);
}

module.exports = {
    getRawData,
    loadResponses,
    getRespond,
    createWordTensor
}