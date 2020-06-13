let net;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

async function app() {
    console.log('Loading mobilenet..');

    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');

    // Create object using TensorFlow.js API
    const webcam = await tf.data.webcam(webcamElement);

    // Read image from the webcam
    const addExample = async classId => {
        // Capture image from webcam
        const img = await webcam.capture();
        const activation = net.infer(img, true);
        classifier.addExample(activation, classId);
        img.dispose();
    }

    // when button clicked
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));

    while (true) {
        if(classifier.getNumClasses() > 0) {
            const img = await webcam.capture();

            // get activation from mobileNet
            const activation = net.infer(img, 'conv_preds');
            const result = await classifier.predictClass(activation);
            const classes = ['A', 'B'];
            document.getElementById('console').innerText = `
                prediction: ${classes[result.label]} \n
                probability: ${result.confidences[result.label]}
            `;
            img.dispose();
        }
        await tf.nextFrame();
  }
}

app();