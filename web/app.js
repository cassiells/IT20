document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-btn');
    const resultSection = document.getElementById('result-section');
    const content = document.getElementById('prediction-content');

    // Sliders
    const workHours = document.getElementById('work_hours');
    const workHoursVal = document.getElementById('work_hours_val');
    const sleepHours = document.getElementById('sleep_hours');
    const sleepHoursVal = document.getElementById('sleep_hours_val');

    workHours.addEventListener('input', () => workHoursVal.textContent = `${workHours.value}h`);
    sleepHours.addEventListener('input', () => sleepHoursVal.textContent = `${sleepHours.value}h`);

    predictBtn.addEventListener('click', async () => {
        // Reset and show initial results container (but hidden content)
        resultSection.classList.add('hidden');

        // Gather Data
        const data = {
            Age: parseInt(document.getElementById('age').value),
            Gender: document.getElementById('gender').value,
            Country: document.getElementById('country').value,
            Job_Role: document.getElementById('job_role').value,
            Experience_Years: parseInt(document.getElementById('experience_years').value),
            Company_Size: document.getElementById('company_size').value,
            Work_Hours_Per_Day: parseFloat(document.getElementById('work_hours').value),
            Meetings_Per_Day: parseInt(document.getElementById('meetings_per_day').value),
            Internet_Speed_Mbps: parseFloat(document.getElementById('internet_speed').value),
            Work_Environment: document.getElementById('work_environment').value,
            Sleep_Hours: parseFloat(document.getElementById('sleep_hours').value),
            Exercise_Hours_Per_Week: parseFloat(document.getElementById('exercise_hours').value),
            Screen_Time_Hours: parseFloat(document.getElementById('screen_time').value),
            Stress_Level: document.getElementById('stress_level').value,
            Productivity_Score: parseInt(document.getElementById('productivity_score').value)
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'API Error');
            }

            const result = await response.json();
            displayResult(result);

            // Show results section and scroll
            resultSection.classList.remove('hidden');
            resultSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Prediction Error:', error);
            alert("AI Engine Error: " + error.message + "\n\nPlease ensure the backend is running.");
        }
    });

    function displayResult(result) {
        const badge = document.getElementById('status-badge');
        const text = document.getElementById('prediction-text');
        const bar = document.getElementById('confidence-bar');
        const confText = document.getElementById('confidence-text');

        const isRisk = result.Burnout_Risk === 'Burnout';

        badge.textContent = isRisk ? 'BURNOUT' : 'CLEAR';
        badge.className = 'status-badge ' + (isRisk ? 'status-danger' : 'status-clear');

        text.textContent = isRisk ? 'Burnout' : 'No Significant Burnout';
        text.style.color = isRisk ? '#ef4444' : '#10b981';

        const confidence = (result.Confidence * 100).toFixed(1);
        bar.style.width = `${confidence}%`;
        confText.textContent = `${confidence}% Confidence Level`;

        // Dynamic bar color
        bar.style.background = isRisk ? '#ef4444' : '#10b981';

        // Ensure content is visible within the card
        content.classList.remove('hidden');
    }
});