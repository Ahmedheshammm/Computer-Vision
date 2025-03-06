document.addEventListener("DOMContentLoaded", function () {
  // Get file input element
  const fileInput = document.getElementById("file");

  // Add event listener to file input
  if (fileInput) {
    fileInput.addEventListener("change", function () {
      const fileName = this.files[0].name;
      const fileSize = (this.files[0].size / 1024 / 1024).toFixed(2);

      // Display file info
      const fileInfo = document.createElement("p");
      fileInfo.textContent = `Selected file: ${fileName} (${fileSize} MB)`;
      fileInfo.style.margin = "10px 0";

      // Remove previous file info if exists
      const previousInfo = document.querySelector(".file-info");
      if (previousInfo) {
        previousInfo.remove();
      }

      // Add class for easy removal later
      fileInfo.classList.add("file-info");

      // Insert after file input
      this.parentNode.appendChild(fileInfo);
    });
  }
});
