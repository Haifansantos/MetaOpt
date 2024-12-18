document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("file");
    const taskTypeSelect = document.getElementById("task_type");
    const labelGroup = document.getElementById("label-group");
    const labelInput = document.getElementById("label");
    const submitButton = document.querySelector("button[type='submit']");

    // Menampilkan nama file yang dipilih
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            alert(`File selected: ${fileInput.files[0].name}`);
        }
    });

    // Validasi sebelum submit
    submitButton.addEventListener("click", (event) => {
        if (!fileInput.files.length) {
            alert("Please upload a dataset file.");
            event.preventDefault();
        }

        if (!labelInput.value && taskTypeSelect.value !== "clustering") {
            alert("Please enter a label column for regression/classification tasks.");
            event.preventDefault();
        }
    });

    // Hide or Show Label Dropdown on Task Type Change
    taskTypeSelect.addEventListener("change", function() {
        const taskType = this.value;
        if (taskType === "clustering") {
            labelGroup.style.display = "none";
            labelInput.removeAttribute("required");
        } else {
            labelGroup.style.display = "block";
            labelInput.setAttribute("required", "required");
        }
    });

    // Initialize State on Load
    window.addEventListener("DOMContentLoaded", function() {
        taskTypeSelect.dispatchEvent(new Event("change"));
    });
});
