const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("fileInput");
const form = document.getElementById("uploadForm");
const loader = document.getElementById("loader");
const videoPreview = document.getElementById("videoPreview");
const previewVideo = document.getElementById("previewVideo");

function showVideoPreview(file) {
  if (file && file.type.startsWith("video/")) {
    const reader = new FileReader();
    reader.onload = (e) => {
      previewVideo.src = e.target.result;
      videoPreview.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
  }
}

dropArea.addEventListener("click", () => fileInput.click());

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  fileInput.files = e.dataTransfer.files;
  if (fileInput.files.length > 0) {
    showVideoPreview(fileInput.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    showVideoPreview(fileInput.files[0]);
  }
});

form.addEventListener("submit", () => {
  loader.classList.remove("hidden");
});
