// Preview uploaded image
document.getElementById("file").addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            document.getElementById("image-preview").src = e.target.result;
            document.getElementById("image-preview").style.display = "block";
        };
        reader.readAsDataURL(file);
    }
});

// Show loading spinner during form submission
document.querySelector("form").addEventListener("submit", function () {
    document.getElementById("loading-spinner").style.display = "block";
});

// Optional: Fetch results dynamically using AJAX
function fetchResults(task, filepath) {
    fetch("/run-inference", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ task: task, filepath: filepath }),
    })
        .then((response) => response.json())
        .then((data) => {
            // Update the page with the results
            console.log(data);
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}