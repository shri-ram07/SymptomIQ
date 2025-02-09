document.addEventListener('DOMContentLoaded', async () => {
    const categorySelector = document.getElementById('category');
    const symptomList = document.getElementById('symptom-list');
    const selectedSymptomsList = document.getElementById('selected-symptoms');
    const predictBtn = document.getElementById('predict-btn');
    const resultSection = document.getElementById('result-section');
    const predictionResult = document.getElementById('prediction-result');

    // Chatbot Elements
    const chatbotInput = document.getElementById('chatbot-input');
    const chatbotMessages = document.getElementById('chatbot-messages');
    const chatbotSend = document.getElementById('chatbot-send');
    const loadingSpinner = document.getElementById('loading-spinner');
    // List of health conditions (41 outputs)
    const HEALTH_CONDITIONS = [
        "Fungal Infection", "Allergy", "Gerd", "Chronic Cholestasis", "Drug Reaction",
        "Peptic Ulcer Disease", "Aids", "Diabetes", "Gastroenteritis", "Bronchial Asthma",
        "Hypertension", "Migraine", "Cervical Spondylosis", "Paralysis (Brain Haemorrhage)",
        "Jaundice", "Malaria", "Chickenpox", "Dengue", "Typhoid", "Hepatitis A", "Hepatitis B",
        "Hepatitis C", "Hepatitis D", "Hepatitis E", "Alcoholic Hepatitis", "Tuberculosis",
        "Common Cold", "Pneumonia", "Dimorphic Haemorrhoids (Piles)", "Heart Attack",
        "Varicose Veins", "Hypothyroidism", "Hyperthyroidism", "Hypoglycaemia", "Osteoarthritis",
        "Arthritis", "Vertigo", "Acne", "Urinary Tract Infection", "Psoriasis", "Impetigo"
    ];
    let selectedSymptoms = [];
    let predictedConditionIndex = null; // To store the predicted condition index

    // Get user's location
    let userLocation = "Unknown Location";
    try {
        const locationData = await getUserLocation();
        if (locationData.latitude && locationData.longitude) {
            userLocation = `${locationData.latitude}, ${locationData.longitude}`;
        }
    } catch (error) {
        console.error(error);
    }

    // Fetch user's location using Geolocation API
    function getUserLocation() {
        return new Promise((resolve, reject) => {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        const latitude = position.coords.latitude;
                        const longitude = position.coords.longitude;
                        resolve({ latitude, longitude });
                    },
                    (error) => {
                        console.error("Error fetching location:", error);
                        resolve({ latitude: null, longitude: null });
                    }
                );
            } else {
                reject("Geolocation is not supported by this browser.");
            }
        });
    }

    // Filter symptoms by category
    categorySelector.addEventListener('change', (e) => {
        const category = e.target.value;
        Array.from(symptomList.children).forEach(item => {
            const itemCategory = item.dataset.category;
            if (category === 'all' || itemCategory === category) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
    });

    // Add symptom to selected list
    symptomList.addEventListener('click', (e) => {
        if (e.target.classList.contains('add-btn')) {
            const symptomItem = e.target.parentElement;
            const symptom = symptomItem.dataset.symptom;

            if (!selectedSymptoms.includes(symptom)) {
                selectedSymptoms.push(symptom);
                updateSelectedSymptoms();
            }
        }
    });

    // Remove symptom from selected list
    selectedSymptomsList.addEventListener('click', (e) => {
        if (e.target.classList.contains('remove-btn')) {
            const symptom = e.target.dataset.symptom;
            selectedSymptoms = selectedSymptoms.filter(s => s !== symptom);
            updateSelectedSymptoms();
        }
    });

    // Update selected symptoms UI
    function updateSelectedSymptoms() {
        selectedSymptomsList.innerHTML = '';
        selectedSymptoms.forEach(symptom => {
            const li = document.createElement('li');
            li.textContent = symptom;
            const removeBtn = document.createElement('button');
            removeBtn.textContent = 'Remove';
            removeBtn.classList.add('remove-btn');
            removeBtn.dataset.symptom = symptom;
            li.appendChild(removeBtn);
            selectedSymptomsList.appendChild(li);
        });
    }

    // Predict health condition
    // Predict health condition
predictBtn.addEventListener('click', async () => {
    if (selectedSymptoms.length === 0) {
        alert('Please select at least one symptom.');
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symptoms: selectedSymptoms })
        });

        const data = await response.json();
        console.log("Backend Response:", data); // Debugging log

        if (data.message) {
            predictionResult.textContent = data.message; // Display the softened message
            resultSection.style.display = 'block';

            // Extract the predicted condition index (if available)
            const conditionMatch = data.message.match(/'([^']+)'/);
            if (conditionMatch) {
                const predictedCondition = conditionMatch[1];
                const conditionIndex = HEALTH_CONDITIONS.indexOf(predictedCondition);
                predictedConditionIndex = conditionIndex !== -1 ? conditionIndex : null;
            }
        } else {
            console.error("Unexpected response format:", data);
            predictionResult.textContent = "An error occurred while processing your request.";
            resultSection.style.display = 'block';
        }
    } catch (error) {
        console.error("Error during prediction:", error);
        predictionResult.textContent = "An error occurred while communicating with the server.";
        resultSection.style.display = 'block';
    }
});

    // Send message to chatbot
    chatbotSend.addEventListener('click', async () => {
        const userMessage = chatbotInput.value.trim();
        if (!userMessage) return;

        // Display user message
        const userMessageElement = document.createElement('p');
        userMessageElement.textContent = `You: ${userMessage}`;
        userMessageElement.style.backgroundColor = '#d1e7dd';
        chatbotMessages.appendChild(userMessageElement);

        // Clear input field
        chatbotInput.value = '';

        // Show loading spinner
        loadingSpinner.style.display = 'flex';

        // Fetch bot response
        try {
            const response = await fetch('/chatbot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userMessage,
                    location: userLocation,
                    symptoms: selectedSymptoms,
                    condition_index: predictedConditionIndex // Pass the predicted condition index
                })
            });

            const data = await response.json();
            const botResponse = data.response;

            // Display bot response
            const botMessageElement = document.createElement('p');
            botMessageElement.textContent = `Bot: ${botResponse}`;
            botMessageElement.style.backgroundColor = '#f8d7da';
            chatbotMessages.appendChild(botMessageElement);

            // Scroll to bottom
            chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        } catch (error) {
            console.error("Error communicating with chatbot:", error);
        } finally {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
        }
    });
});