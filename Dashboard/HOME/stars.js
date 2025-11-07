// ============================
// Stars Background Animation
// ============================
const canvas = document.getElementById('stars');
if (canvas) {
    const ctx = canvas.getContext('2d');

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const stars = Array.from({ length: 150 }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        radius: Math.random() * 1.5,
        velocity: Math.random() * 0.5 + 0.2
    }));

    function animateStars() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "white";
        stars.forEach(star => {
            ctx.beginPath();
            ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
            ctx.fill();
            star.y += star.velocity;
            if (star.y > canvas.height) {
                star.y = 0;
                star.x = Math.random() * canvas.width;
            }
        });
        requestAnimationFrame(animateStars);
    }
    animateStars();
}


//======================
//Random articles fetch
//======================
// Array of articles with images and links
const articles = [

        {
            title: "Climate Change Impact on Agriculture",
            content: "Explore how climate change affects global food production and sustainability.",
            image: "https://images.unsplash.com/photo-1501004318641-b39e6451bec6", 
            link: "https://www.epa.gov/climateimpacts/climate-change-impacts-agriculture-and-food-supply"
        },
        {
            title: "Renewable Energy Solutions",
            content: "Learn about how renewable technologies combat climate change and promote sustainability.",
            image: "https://images.unsplash.com/photo-1505245208761-ba872912fac0", 
            link: "https://www.nrdc.org/bio/noah-long/renewable-energy-key-fighting-climate-change"
        },
        {
            title: "Global Warming and Rising Sea Levels",
            content: "Understand how rising oceans threaten coastal cities and natural ecosystems.",
            image: "https://images.unsplash.com/photo-1599423300746-b62533397364", 
            link: "https://www.climate.gov/news-features/understanding-climate/climate-change-global-sea-level"
        },
        {
            title: "The Role of Forests in Mitigating Climate Change",
            content: "Discover how forests are crucial for carbon sequestration and combating global warming.",
            image: "../IMAGES/Forest.png​", 
            link: "https://www.wwf.org.uk/what-we-do/forests"
        },
        {
            title: "The Importance of Reducing Carbon Footprint",
            content: "Understand the impact of reducing carbon emissions and the role of individuals and industries.",
            image: "../IMAGES/Carbon_footprint.png", 
            link: "https://www.dbs.com/digibank/in/articles/lifestyle/importance-of-reducing-carbon-footprint"
        },
        {
            title: "The Impact of Climate Change on Biodiversity",
            content: "Learn about the threats climate change poses to species and ecosystems worldwide.",
            image: "../IMAGES/Biodiversity.jpg​", 
            link: "https://www.iucn.org/resources/issues-briefs/climate-change-and-biodiversity"
        },
        {
            title: "Ocean Acidification: The Hidden Consequence of Climate Change",
            content: "Explore how the increasing acidity in oceans affects marine life and ecosystems.",
            image: "../IMAGES/Ocean.jpeg",
            link: "https://www.pmel.noaa.gov/co2/story/Ocean+Acidification"
        },
        {
            title: "The Economic Costs of Climate Change",
            content: "Examine the financial burden of climate change and the importance of sustainable development.",
            image: "../IMAGES/Economic_cost.jpg", 
            link: "https://www.e-education.psu.edu/earth103/node/717"
        }
];

// Function to insert random articles dynamically
function displayRandomArticles() {
    // Shuffle articles array to get random order
    const shuffledArticles = articles.sort(() => 0.5 - Math.random());

    // Take the first 3 shuffled articles
    const selectedArticles = shuffledArticles.slice(0, 3);

    // Insert articles into the container
    const container = document.getElementById('articles-container');
    selectedArticles.forEach(article => {
        const articleHTML = `
            <div class="col-md-4">
                <div class="card h-100">
                    <img src="${article.image}" class="card-img-top" alt="${article.title}">
                    <div class="card-body d-flex flex-column">
                        <h3 class="card-title">${article.title}</h3>
                        <p class="card-text">${article.content}</p>
                        <a href="${article.link}" target="_blank" class="btn btn-primary mt-auto">Read More</a>
                    </div>
                </div>
            </div>
        `;
        container.innerHTML += articleHTML;
    });
}

// Load random articles when the page is loaded
document.addEventListener('DOMContentLoaded', displayRandomArticles);



// ============================
// Geolocation and Weather API Fetch
// ============================
navigator.geolocation.getCurrentPosition(success => {
    const lat = success.coords.latitude;
    const lon = success.coords.longitude;

    // Fetch weather from OpenWeatherMap
    fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=69cdfb708fbc7f6b1523ca4e251af9a7&units=metric`)
        .then(res => res.json())
        .then(data => {
            // Update the temperature, icon, and description
            const temp = data.main.temp.toFixed(1);
            const weatherIcon = `https://openweathermap.org/img/wn/${data.weather[0].icon}.png`;
            const weatherDesc = data.weather[0].description;
            const humidity = data.main.humidity;
            const feelsLike = data.main.feels_like.toFixed(1);
            const precipitation = data.weather[0].main === 'Rain' ? data.rain ? data.rain["1h"] : 0 : 0; // Precipitation in last 1 hour

            // Insert data into HTML
            document.getElementById("temp").textContent = `${temp} °C`;
            document.getElementById("city-label").textContent = data.name;
            document.getElementById("weather-icon").src = weatherIcon;
            document.getElementById("weather-desc").textContent = weatherDesc.charAt(0).toUpperCase() + weatherDesc.slice(1); // Capitalize first letter
            document.getElementById("humidity").textContent = `Humidity: ${humidity} %`;
            document.getElementById("feels-like").textContent = `Feels Like: ${feelsLike} °C`;
            document.getElementById("precipitation").textContent = `Precipitation: ${precipitation} mm`;
        })
        .catch(err => {
            console.error("Weather API error:", err);
            document.getElementById("temp").textContent = "N/A";
            document.getElementById("city-label").textContent = "N/A";
            document.getElementById("weather-desc").textContent = "N/A";
            document.getElementById("humidity").textContent = "Humidity: N/A";
            document.getElementById("feels-like").textContent = "Feels Like: N/A";
            document.getElementById("precipitation").textContent= "Precipitation: N/A";
        });

    // Fetch PM2.5 and PM10 data from WAQI API
    fetch(`https://api.waqi.info/feed/geo:${lat};${lon}/?token=43aac9a61aa4653d7094c54b3ae4ff74d320c9f7`)
        .then(res => res.json())
        .then(data => {
            // Check if the data is valid
            if (data.status === "ok" && data.data) {
                const pm25 = data.data.iaqi.pm25 ? data.data.iaqi.pm25.v : "N/A";
                const pm10 = data.data.iaqi.pm10 ? data.data.iaqi.pm10.v : "N/A";

                document.getElementById("pm25").textContent = `${pm25} µg/m³`;
                document.getElementById("pm10").textContent = `${pm10} µg/m³`;
            } else {
                document.getElementById("pm25").textContent = "N/A";
                document.getElementById("pm10").textContent = "N/A";
            }
        })
        .catch(err => {
            console.error("Air Quality API error:", err);
            document.getElementById("pm25").textContent = "N/A";
            document.getElementById("pm10").textContent = "N/A";
        });

}, error => {
    console.error("Geolocation error:", error);
    document.getElementById("city-label").textContent = "(Location unavailable)";
    document.getElementById("temp").textContent = "N/A";
    document.getElementById("weather-desc").textContent = "N/A";
    document.getElementById("humidity").textContent = "Humidity: N/A";
    document.getElementById("feels-like").textContent = "Feels Like: N/A";
    document.getElementById("precipitation").textContent = "Precipitation: N/A";
    document.getElementById("pm25").textContent = "N/A";
    document.getElementById("pm10").textContent = "N/A";
});


// Fetch a random climate tip (You can replace this with any dynamic source you prefer)
const climateTips = [
    "🌱 Use public transport or cycle to reduce your carbon footprint and improve air quality.",
    "🌍 Reduce water consumption to conserve resources and help fight climate change.",
    "🌿 Plant trees to absorb CO2 and improve air quality.",
    "🌡️ Switch off electrical appliances when not in use to save energy.",
    "💨 Use energy-efficient appliances to reduce your carbon footprint.",
    "🌱 Reduce your carbon footprint by switching to energy-efficient appliances.",
    "🌞 Support clean energy initiatives like wind, solar, and geothermal power.",
    "♻️ Reduce, reuse, and recycle to minimize waste and conserve resources.",
    "📚 Educate others about climate change and its impacts."
];

// Function to generate a random tip
function getClimateTip() {
    const randomTip = climateTips[Math.floor(Math.random() * climateTips.length)];
    document.getElementById("climate-tip").textContent = randomTip;
}

// Call the function to display a random climate tip when the page loads
window.addEventListener('load', getClimateTip);

/*
//  data for the temperature trend (you can replace this with real data)*/

navigator.geolocation.getCurrentPosition(success => {
    const lat = success.coords.latitude;
    const lon = success.coords.longitude;

    const apiUrl = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=temperature_2m&timezone=auto`;

    fetch(apiUrl)
        .then(response => response.json())
        .then(data => {
            const times = data.hourly.time.slice(0, 12);
            const temperatures = data.hourly.temperature_2m.slice(0, 12);

            const ctx = document.getElementById('tempTrendChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: times.map(time => new Date(time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })),
                    datasets: [{
                        label: 'Temperature (°C)',
                        data: temperatures,
                        borderColor: '#00c3ff',
                        backgroundColor: 'rgba(0, 195, 255, 0.2)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error fetching weather data:', error));
});

// ============================
// Preloader Functionality
// ============================
window.addEventListener('load', () => {
    const preloader = document.getElementById('preloader');
    preloader.style.opacity = '0';
    setTimeout(() => {
        preloader.style.display = 'none';
    }, 1000);
});

// ============================
// Scroll to Top Button Functionality
// ============================
const scrollTopBtn = document.getElementById('scrollTopBtn');

if (scrollTopBtn) {
    window.onscroll = function () {
        if (document.body.scrollTop > 200 || document.documentElement.scrollTop > 200) {
            scrollTopBtn.style.display = "block";
        } else {
            scrollTopBtn.style.display = "none";
        }
    };

    scrollTopBtn.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    });
}
