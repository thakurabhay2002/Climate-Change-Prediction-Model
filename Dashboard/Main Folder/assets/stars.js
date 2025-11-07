function initStars(canvas) {
    const ctx = canvas.getContext("2d");
    let stars = [];

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    function createStars(count) {
        stars = [];
        for (let i = 0; i < count; i++) {
            stars.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                radius: Math.random() * 1.5 + 0.5,
                dx: (Math.random() - 0.5) * 0.5,
                dy: (Math.random() - 0.5) * 0.5,
            });
        }
    }

    function drawStars() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "white";
        for (let star of stars) {
            ctx.beginPath();
            ctx.arc(star.x, star.y, star.radius, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    function updateStars() {
        for (let star of stars) {
            star.x += star.dx;
            star.y += star.dy;
            if (star.x < 0 || star.x > canvas.width) star.dx *= -1;
            if (star.y < 0 || star.y > canvas.height) star.dy *= -1;
        }
    }

    function animate() {
        updateStars();
        drawStars();
        requestAnimationFrame(animate);
    }

    createStars(120);
    animate();
}

// --- Detect canvas injection dynamically ---
const observer = new MutationObserver(() => {
    const canvas = document.getElementById("star-canvas");
    if (canvas && !canvas.dataset.animated) {
        canvas.dataset.animated = "true";
        initStars(canvas);
    }
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});
