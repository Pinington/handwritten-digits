const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let drawing = false;

$("#canvas")
  .on("mousedown", function (e) {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
  })
  .on("mouseup mouseleave", function () {
    drawing = false;
  })
  .on("mousemove", function (e) {
    if (!drawing) return;

    ctx.lineWidth = 6;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";

    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  });


$("#clear-canvas").on("click", function() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
})

function sendCanvas() {
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data; // RGBA
    const gray = [];

    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        // grayscale value 0–255
        const value = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
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