const tf = require('@tensorflow/tfjs-node');
const fileIO = require('@tensorflow/tfjs-node/dist/io/file_system');
const fs = require('fs');
const path = require('path');

const stemmer = require('./stemmer_module');
const tokenize = require('./tokenize_module');

let CURRENT_DIR;
let TRAIN_WORDS = [];
let TRAIN_CLASSES = [];
const TRAIN_DOCS = [];
const TRAIN_RESPONSES = [];
const IGNORE_WORDS = ['?'];
let tokenWord = [];
let TRAINING_DATA_TEMP = [];
let TRAINING_DATA = [];
let TRAIN_X = [];
let TRAIN_Y = [];
let TF_MODEL;

function initializeTrainingData(dataUrl, savedir) {
    let RAW_DATA = fs.readFileSync(dataUrl);
    let JSON_DATA = JSON.parse(RAW_DATA);
    CURRENT_DIR = savedir;

    for (let i in JSON_DATA.intents) {
        const intents = JSON_DATA.intents[i];
        for (let j in intents.patterns) {
            const patterns = intents.patterns[j];
            let tempTokens = tokenize(patterns);
            let tokenized_words = [];
            for (let i in tempTokens) {
                tokenized_words.push(stemmer(tempTokens[i].token));
            };
            tokenWord.push(tokenized_words);
            TRAIN_DOCS.push([tokenized_words, intents.tag]);

            if (!TRAIN_CLASSES.includes(intents.tag)) {
                TRAIN_CLASSES.push(intents.tag);
                const responsesA = [intents.tag, intents.responses];
                TRAIN_RESPONSES.push(responsesA);
            }
        }
    }

    for (let b in tokenWord) {
        for (let i in tokenWord[b]) {
            if (!tokenWord[b][i].includes(IGNORE_WORDS)) {
                TRAIN_WORDS.push(stemmer(tokenWord[b][i].toLowerCase()));
            }
        }
    }

    TRAIN_WORDS = Array.from(new Set(TRAIN_WORDS.sort()));
    TRAIN_CLASSES = Array.from(new Set(TRAIN_CLASSES.sort()));

    runTraining();
}

async function getTrainingData() {
    for (let dc in TRAIN_DOCS) {
        let output_row = tf.zeros([1, TRAIN_CLASSES.length]).dataSync();
        let bag_words = [];
        let docs = TRAIN_DOCS[dc];
        let word_patterns = [];

        for (let b in docs[0]) {
            word_patterns.push(stemmer(docs[0][b].toLowerCase()));
        }

        for (let i of TRAIN_WORDS.entries()) {
            //if (i[0] == 0) { console.log(i[1], word_patterns) };
            if (word_patterns.includes(i[1])) {
                bag_words[i[0]] = 1;
            } else {
                bag_words[i[0]] = 0;
            }

        }
        const classIndex = TRAIN_CLASSES.indexOf(docs[1]);
        output_row[classIndex] = 1;
        TRAINING_DATA_TEMP.push([bag_words, output_row]);
    }

    await tf.data.array(TRAINING_DATA_TEMP).shuffle(3).forEachAsync(eData => {
        TRAINING_DATA.push(eData);
    });
}

async function runTraining() {
    await getTrainingData();

    for (let i in TRAINING_DATA) {
        TRAIN_X.push(TRAINING_DATA[i][0]);
        TRAIN_Y.push(TRAINING_DATA[i][1]);
    }

    TF_MODEL = tf.sequential();
    TF_MODEL.add(tf.layers.dense({ units: 128, inputShape: [TRAIN_X[0].length, ], activation: 'relu' }));
    TF_MODEL.add(tf.layers.dropout({ rate: 0.5 }));
    TF_MODEL.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    TF_MODEL.add(tf.layers.dropout({ rate: 0.5 }));
    TF_MODEL.add(tf.layers.dense({ units: TRAIN_Y[0].length, activation: 'softmax' }));

    TF_MODEL.compile({ loss: 'categoricalCrossentropy', optimizer: tf.train.sgd(0.1), metrics: ['acc'] });

    const bnlp_models = path.resolve(CURRENT_DIR, './bnlp_models/');
    fs.mkdir(bnlp_models, { recursive: true }, err => {
        if (err !== null) {
            console.log('\x1b[31mMaking Directory Error: ' + err + '\x1b[0m');
        }
    });

    await TF_MODEL.fit(tf.tensor(TRAIN_X), tf.tensor(TRAIN_Y), {
        batchSize: 5,
        epochs: 200,
        verbose: 0,
        callbacks: {
            onEpochEnd: async(epoch, logs) => {
                console.log('\x1b[0m\x1b[34mEpoch: ' + (epoch + 1) + '\x1b[35m | Loss: ' + logs.loss.toFixed(5) +
                    '\x1b[33m | Accuracy: ' + logs.acc.toFixed(5));
            }
        }
    }).then(info => {
        const infoIndex = info.epoch.length - 1;
        const finalLoss = info.history.loss[infoIndex].toFixed(5);
        const finalAcc = info.history.acc[infoIndex].toFixed(5);
        console.log('\x1b[0m\x1b[37m\x1b[44m BuddhiNLP \x1b[0m', '\x1b[32m Training Completed at \x1b[0m' +
            '==>\x1b[0m\x1b[35m Loss: ' + finalLoss + '\x1b[33m | Accuracy: ' + finalAcc + '\x1b[0m');
    });

    const timeStamp = Date.now();
    const modelOutFolder = path.resolve(bnlp_models, timeStamp + '/');

    fs.mkdir(modelOutFolder, { recursive: true }, err => {
        if (err !== null) {
            console.log('\x1b[31mMaking Directory Error: ' + err + '\x1b[0m');
        }
    });
    await TF_MODEL.save(fileIO.fileSystem(modelOutFolder));

    const metaOutPath = path.resolve(modelOutFolder, 'model_metadata.json');
    const metadataStr = JSON.stringify({ 'words': TRAIN_WORDS, 'classes': TRAIN_CLASSES, 'responses': TRAIN_RESPONSES });
    fs.writeFileSync(metaOutPath, metadataStr, { encoding: 'utf8' });

    console.log('\x1b[0m\x1b[37m\x1b[44m BuddhiNLP \x1b[0m',
        '\x1b[32m Model saved at \x1b[36m' + modelOutFolder + '\x1b[0m');
}

module.exports = {
    initializeTrainingData
};