//Imports
const tf = require('@tensorflow/tfjs-node');
const { getRawData, loadResponses, getRespond, createWordTensor } = require('./modules/classify_module');
const { initializeTrainingData } = require('./modules/train_module');
//Global variables
const buddhi = () => {
    console.log('\x1b[0m\x1b[37m\x1b[44m BuddhiNLP \x1b[0m');
};
let TF_MODEL;
let CLASSIFIER_DATA;
let MODEL_WORDS;
let MODEL_CLASSES;
let MODEL_RESPONSES = [];

//Load model
buddhi.loadModel = async(modelUrl, dataUrl, callBackFunction) => {
    if (modelUrl == "") {
        throw "Invalid Model Url. (BNLPE001)";
    }
    if (dataUrl == "") {
        throw "Invalid Data Url. (BNLPE002)";
    }
    if (!callBackFunction) {
        throw "Invalid Call back function. (BNLPE003)";
    }
    TF_MODEL = await tf.loadLayersModel(modelUrl);
    CLASSIFIER_DATA = getRawData(dataUrl);
    MODEL_WORDS = CLASSIFIER_DATA['words'];
    MODEL_CLASSES = CLASSIFIER_DATA['classes'];
    MODEL_RESPONSES = loadResponses(CLASSIFIER_DATA['responses']);
    //console.log('%c BuddhiNLP ', 'background:#f7931e; color:#fff', 'Model Initialized');
    console.log('\x1b[0m\x1b[37m\x1b[44m BuddhiNLP \x1b[0m', '\x1b[0m Model Loaded!')
    callBackFunction();
    return TF_MODEL;
};

//Bot Talk
buddhi.botTalk = async(callBackFunction) => {
    if (!callBackFunction) {
        throw "Invalid Call back function. (BNLPE003)";
    }
    let modelUrl = 'https://api.buddhilive.com/model.json';
    let dataUrl = 'https://api.buddhilive.com/model_metadata.json'
    TF_MODEL = await tf.loadLayersModel(modelUrl);
    CLASSIFIER_DATA = getRawData(dataUrl);
    MODEL_WORDS = CLASSIFIER_DATA['words'];
    MODEL_WORDS = CLASSIFIER_DATA['words'];
    MODEL_CLASSES = CLASSIFIER_DATA['classes'];
    MODEL_RESPONSES = loadResponses(CLASSIFIER_DATA['responses']);
    console.log('\x1b[0m\x1b[37m\x1b[44m BuddhiNLP \x1b[0m', '\x1b[32m BotTalk Initialized!')
    callBackFunction();
    return TF_MODEL;
};

//Classify sentences
buddhi.classify = (USER_INPUT) => {
    //const ERROR_THRESHOLD = 0.25
    if (!TF_MODEL) {
        throw "Model not initialized. (BNLPE004)";
    }
    if (USER_INPUT == "") {
        throw "No user input. (BNLPE005)";
    }

    const bow_data = createWordTensor(USER_INPUT, MODEL_WORDS);

    const input_data = tf.stack([bow_data]);

    let classified_results = TF_MODEL.predict(input_data);
    //Get results
    classified_results = classified_results.dataSync();
    const classIndex = Math.max(...classified_results);
    classified_results = classified_results.indexOf(classIndex);
    const botResponse = getRespond(MODEL_CLASSES[classified_results]);

    return [botResponse, MODEL_CLASSES[classified_results], classIndex];
}

//Train a model
buddhi.train = (dataUrl, savedir) => {
    if (dataUrl == "") {
        throw "Invalid url provided for training data. (BNLPE006)";
    }
    if (savedir == "") {
        throw "Invalid directory path. (BNLPE007)";
    }
    initializeTrainingData(dataUrl, savedir);
}

module.exports = buddhi;