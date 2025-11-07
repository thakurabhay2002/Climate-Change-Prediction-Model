document.addEventListener("DOMContentLoaded", function () {
    const headers = document.querySelectorAll(".toggle-header");

    headers.forEach(header => {
        header.addEventListener("click", () => {
            const body = header.nextElementSibling;
            body.classList.toggle("show");

            // Optional: Toggle arrow icon
            header.textContent = header.textContent.includes("🔼")
                ? header.textContent.replace("🔼", "🔽")
                : header.textContent.replace("🔽", "🔼");
        });
    });
});
document.addEventListener("DOMContentLoaded", function () {
    const headers = document.querySelectorAll(".toggle-header");

    headers.forEach(header => {
        header.addEventListener("click", function () {
            const body = document.getElementById(header.id.replace("-header", "-body"));
            if (body.classList.contains("show")) {
                body.classList.remove("show");
            } else {
                body.classList.add("show");
            }
        });
    });
});
