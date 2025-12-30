document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const btn = document.getElementById('predictBtn');
    const loader = document.getElementById('loader');
    const btnText = btn.querySelector('span');
    const resultCard = document.getElementById('result');

    // UI Loading State
    btn.disabled = true;
    loader.style.display = 'block';
    btnText.style.opacity = '0';
    resultCard.classList.add('hidden');

    // Gather data
    const formData = new FormData(e.target);
    const data = {
        Gender: formData.get('Gender'),
        Age: parseInt(formData.get('Age')),
        Height: parseFloat(formData.get('Height')),
        Weight: parseFloat(formData.get('Weight')),
        Duration: parseFloat(formData.get('Duration')),
        Heart_Rate: parseFloat(formData.get('Heart_Rate')),
        Body_Temp: parseFloat(formData.get('Body_Temp'))
    };

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const result = await response.json();

        // Update UI with result
        const calories = result.calories_burnt.toFixed(1);

        // Animate number
        resultCard.classList.remove('hidden');
        animateValue(document.getElementById('caloriesValue'), 0, parseFloat(calories), 1000);

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction. Please make sure the API is running.');
    } finally {
        // Reset UI State
        btn.disabled = false;
        loader.style.display = 'none';
        btnText.style.opacity = '1';
    }
});

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = (progress * (end - start) + start).toFixed(1);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
