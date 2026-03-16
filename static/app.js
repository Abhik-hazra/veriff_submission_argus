document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("dropzone");
    const input = document.getElementById("videos");
    const fileStatus = document.getElementById("file-status");

    if (!dropzone || !input || !fileStatus) {
        return;
    }

    ["dragenter", "dragover", "drop"].forEach((eventName) => {
        window.addEventListener(eventName, (e) => {
            if (!dropzone.contains(e.target)) {
                e.preventDefault();
            }
        });
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.add("drag-active");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.remove("drag-active");
        });
    });

    input.addEventListener("change", () => {
        updateFileStatus(input.files);
    });

    function updateFileStatus(files) {
        if (!files || files.length === 0) {
            fileStatus.textContent = "No files selected yet.";
            fileStatus.className = "file-status";
            return;
        }

        const names = Array.from(files).map((f) => f.name);
        fileStatus.textContent = `${files.length} file(s) selected: ${names.join(", ")}`;
        fileStatus.className = "file-status success";
    }
});
