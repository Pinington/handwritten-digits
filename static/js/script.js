const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let drawing = false;

// Helper: get coordinates for mouse or touch
function getPos(e) {
    if (e.touches) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: e.touches[0].clientX - rect.left,
            y: e.touches[0].clientY - rect.top
        };
    } else {
        return {
            x: e.offsetX,
            y: e.offsetY
        };
    }
}

// Start drawing
function startDraw(e) {
    drawing = true;
    ctx.beginPath();
    const pos = getPos(e);
    ctx.moveTo(pos.x, pos.y);
    e.preventDefault();
}

// Stop drawing
function stopDraw(e) {
    drawing = false;
    e.preventDefault();
}

// Draw
function draw(e) {
    if (!drawing) return;
    const pos = getPos(e);

    ctx.lineWidth = 6;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";

    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    e.preventDefault();
}

// Mouse events
$("#canvas")
  .on("mousedown", startDraw)
  .on("mousemove", draw)
  .on("mouseup mouseleave", stopDraw);

// Touch events
$("#canvas")
  .on("touchstart", startDraw)
  .on("touchmove", draw)
  .on("touchend touchcancel", stopDraw);

// Clear canvas
$("#clear-canvas").on("click", function() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// Prevent scrolling when touching the canvas
canvas.addEventListener("touchstart", e => e.preventDefault(), { passive: false });
canvas.addEventListener("touchmove", e => e.preventDefault(), { passive: false });

function sendCanvas() {
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data; // RGBA
    const gray = [];

    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        // grayscale value 0–255 normalized then made binary
        const value = Math.round(Math.round(0.299 * r + 0.587 * g + 0.114 * b) / 255);
        gray.push(value);
    }

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            width: canvas.width,
            height: canvas.height,
            pixels: gray
        })
    })
    .then(res => res.json())
    .then(data => {
        console.log("Prediction:", data.prediction);
        $("#result").text("Result is " + data.prediction)
    });
}