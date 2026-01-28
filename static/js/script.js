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

    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";

    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  });


$("#clear-canvas").on("click", function() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
})