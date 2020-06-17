const NUM_CLASSES = 2;
const IMAGE_SIZE = 227;
const TOPK = 10;

const classes = ['First', 'Second'];
let testPrediction = false;
let training = true;
let video = document.getElementById("webcam");

class App {
    constructor() {
        this.infoTexts = [];
        this.training = -1 //no class being trained
        this.recordSamples = false;

        // initiate deeplearn.js and KNN classifier
        this.loadClassifierAndModel();
        this.initiateWebcam();
        this.setupButtonEvents();
    }

    async loadClassifierAndModel() {
        this.knn = knnClassifier.create();
        this.mobilenetModule = await mobilenet.load();
        console.log("model loaded");
        this.start();
    }

    initiateWebcam() {
        navigator.mediaDevices
            .getUserMedia({ video: true, audio:false})
            .then(stream => {
                video.srcObject = stream;
                video.width = IMAGE_SIZE;
                video.height = IMAGE_SIZE;
            });
    }

    setupButtonEvents() {
        for (let i = 0; i < NUM_CLASSES; i++) {
          let button = document.getElementsByClassName("button")[i];
    
          button.onmousedown = () => {
            this.training = i;
            this.recordSamples = true;
          };
          button.onmouseup = () => (this.training = -1);
    
          const infoText = document.getElementsByClassName("info-text")[i];
          infoText.innerText = " No examples added";
          this.infoTexts.push(infoText);
        }
    }

    start() {
        if (this.timer) {
          this.stop();
        }
        this.timer = requestAnimationFrame(this.animate.bind(this));
    }
    
    stop() {
        cancelAnimationFrame(this.timer);
    }
    
    async animate() {
        if (this.recordSamples) {
            const image = tf.browser.fromPixels(video);
            let logits;
            const infer = () => this.mobilenetModule.infer(image, 'conv_preds');
            if (this.training != -1) {
                logits = infer();
                this.knn.addExample(logits, this.training);
            }
            const numClasses = this.knn.getNumClasses();
            if (testPrediction) {
                training = false;
                if (numClasses > 0) {
                    logits = infer();
                    const res = await this.knn.predictClass(logits, TOPK);
                    for (let i=0; i<NUM_CLASSES; i++) {
                        const exampleCount = this.knn.getClassExampleCount();
                        if (res.classIndex == i) {
                            this.infoTexts[i].style.fontWeight = 'bold';
                        } else {
                            this.infoTexts[i].style.fontWeight = 'normal';
                        }
                        if (exampleCount[i] > 0) {
                            this.infoTexts[i].innerText = ` ${
                                exampleCount[i]
                            } 
                            examples - ${res.confidences[i] * 100}%`;
                        }
                    }
                }
            }

            if (training) {
                // The number of examples for each class
                const exampleCount = this.knn.getClassExampleCount();
        
                for (let i = 0; i < NUM_CLASSES; i++) {
                  if (exampleCount[i] > 0) {
                    this.infoTexts[i].innerText = ` ${exampleCount[i]} examples`;
                  }
                }
            }

            image.dispose();
            if (logits != null) {
                logits.dispose();
                }
        }
        this.timer = requestAnimationFrame(this.animate.bind(this));
    }
}

document
  .getElementsByClassName("test-predictions")[0]
  .addEventListener("click", function() {
    testPrediction = true;
  });

new App();