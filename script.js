const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const predictionsParagraph = document.getElementById('predictions');
const availableClasses = document.getElementById('availableClasses');
const classList = document.getElementById('classList');
const classImagesGrid = document.getElementById('classImagesGrid');

const classNames = [
    'Charmander', 'Bulbasaur', 'Squirtle', 'Eevee', 'Pikachu',
    'Snorlax', 'Gengar', 'Gyarados', 'Dragonite', 'Jigglypuff'
];

const classImages = [
    'images/Charmander.jpg', 'images/Bulbasaur.jpg', 'images/Squirtle.jpg', 'images/Eevee.jpg', 'images/Pikachu.jpg',
    'images/Snorlax.jpg', 'images/Gengar.jpg', 'images/Gyarados.jpg', 'images/Dragonite.jpg', 'images/Jigglypuff.jpg'
];

let model;

// Fonction pour remplir la grille avec les images des Pokémon
function populateClassImagesGrid() {
    classImages.forEach((imageSrc, index) => {
        const img = document.createElement('img');
        img.src = imageSrc;
        img.alt = classNames[index];
        img.title = classNames[index];
        classImagesGrid.appendChild(img);
    });
}

// Appelez la fonction pour afficher les images
populateClassImagesGrid();

// Charger le modèle
tf.loadGraphModel('assets/model.json').then(loadedModel => {
    model = loadedModel;
    console.log('Custom model loaded successfully');
});

imageInput.addEventListener('change', event => {
    const file = event.target.files[0];
    if (file) {
        if (file.type !== 'image/jpeg') {
            alert('Please upload only JPG images.');
            return;
        }

        const reader = new FileReader();
        reader.onload = () => {
            previewImage.src = reader.result;
            previewImage.onload = () => classifyImage(previewImage);
        };
        reader.readAsDataURL(file);
    }
});

async function classifyImage(imgElement) {
    if (!model) {
        predictionsParagraph.innerText = "Model is not loaded yet. Please wait...";
        return;
    }

    predictionsParagraph.innerText = "Classifying...";

    const tensor = tf.browser.fromPixels(imgElement)
        .resizeBilinear([177, 177])
        .toFloat()
        .div(tf.scalar(255));

    const cropHeight = 128;
    const cropWidth = 128;
    const startHeight = Math.floor((tensor.shape[0] - cropHeight) / 2);
    const startWidth = Math.floor((tensor.shape[1] - cropWidth) / 2);

    const croppedTensor = tensor.slice([startHeight, startWidth, 0], [cropHeight, cropWidth, 3]);

    const normalizedTensor = croppedTensor.sub(tf.tensor([0.485, 0.456, 0.406]))
        .div(tf.tensor([0.229, 0.224, 0.225]));

    const batchedTensor = normalizedTensor.expandDims(0);

    const predictions = await model.predict(batchedTensor);
    const predictionData = await predictions.data();

    // Masquer la grille des images disponibles
    availableClasses.style.display = 'none';
    classImagesGrid.style.display = 'none';

    displayPredictions(predictionData);
}


function applySoftmax(predictions) {
    const expValues = predictions.map(x => Math.exp(x));
    const sumExp = expValues.reduce((sum, x) => sum + x, 0);
    return expValues.map(x => x / sumExp);
}

function displayPredictions(predictionData) {
    predictionsParagraph.innerHTML = "<strong>Predictions:</strong><br>";

    const predictionDataSoftmax = applySoftmax(predictionData);

    const predictionsWithClasses = Array.from(predictionDataSoftmax)
        .map((probability, index) => ({ classIndex: index, probability }))
        .sort((a, b) => b.probability - a.probability);

    predictionsWithClasses.forEach(prediction => {
        const className = classNames[prediction.classIndex] || `Class ${prediction.classIndex}`;
        const probabilityPercentage = Math.max(prediction.probability, 0) * 100;

        const predictionRow = document.createElement('div');
        predictionRow.className = 'prediction-row';

        const pokemonImage = document.createElement('img');
        pokemonImage.src = classImages[prediction.classIndex];
        pokemonImage.alt = className;
        pokemonImage.className = 'prediction-image';

        const classInfo = document.createElement('span');
        classInfo.innerText = `${className}: ${probabilityPercentage.toFixed(2)}%`;

        const progressBarContainer = document.createElement('div');
        progressBarContainer.className = 'progress-container';

        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        progressBar.style.width = '0%';

        progressBarContainer.appendChild(progressBar);
        predictionRow.appendChild(pokemonImage);
        predictionRow.appendChild(classInfo);
        predictionRow.appendChild(progressBarContainer);
        predictionsParagraph.appendChild(predictionRow);

        animateProgressBar(progressBar, probabilityPercentage);
    });
}

function animateProgressBar(progressBar, targetWidth) {
    let currentWidth = 0;
    const interval = setInterval(() => {
        if (currentWidth < targetWidth) {
            currentWidth += 2;
            progressBar.style.width = `${currentWidth}%`;
        } else {
            clearInterval(interval);
        }
    }, 10);
}
