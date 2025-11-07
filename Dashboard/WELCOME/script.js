var canvas = document.getElementById("stars"),
    ctx = canvas.getContext("2d"),
    stars = [],
    numStars = 200;

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Generate stars
for (var i = 0; i < numStars; i++) {
    stars.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        radius: Math.random() * 1.5,
        velocityX: (Math.random() - 0.5) * 0.5,
        velocityY: (Math.random() - 0.5) * 0.5
    });
}

// Draw and Move stars
function drawStars() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    for (var i = 0; i < stars.length; i++) {
        var s = stars[i];
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.radius, 0, 2 * Math.PI);
        ctx.fill();
    }
    moveStars();
}

function moveStars() {
    for (var i = 0; i < stars.length; i++) {
        var s = stars[i];
        s.x += s.velocityX;
        s.y += s.velocityY;
        if (s.x < 0 || s.x > canvas.width) s.velocityX = -s.velocityX;
        if (s.y < 0 || s.y > canvas.height) s.velocityY = -s.velocityY;
    }
}

function animateStars() {
    drawStars();
    requestAnimationFrame(animateStars);
}

animateStars();

// Resize canvas if window size changes
window.addEventListener('resize', function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});
