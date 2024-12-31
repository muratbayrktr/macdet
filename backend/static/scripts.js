document.addEventListener("DOMContentLoaded", () => {
    // Define testbeds as JSON if it's passed dynamically
    console.log("Testbeds:", testbeds);

    // Toggle between sampling and manual input
    function toggleInputMode() {
        const inputMode = document.getElementById("input-mode").value;
        const samplingOptions = document.getElementById("sampling-options");
        const manualInput = document.getElementById("manual-input");
        const sampledDataOutput = document.getElementById("sampled-data-output");

        if (inputMode === "manual") {
            samplingOptions.style.display = "none";
            manualInput.style.display = "block";
            sampledDataOutput.style.display = "none";
        } else {
            samplingOptions.style.display = "block";
            sampledDataOutput.style.display = "block";
            manualInput.style.display = "none";
        }

        
    }
    window.toggleInputMode = toggleInputMode;

    // Populate testbed options dynamically
    function populateTestbedOptions() {
        const testbedDropdown = document.getElementById('testbed');
        const selectedTestbed = testbedDropdown.value;

        const testbedDetails = Object.values(testbeds).flat().find(t => t.testbed_name === selectedTestbed);

        if (testbedDetails && testbedDetails.subtypes) {
            const testbedTypeDropdown = document.getElementById('testbed-type');
            testbedTypeDropdown.innerHTML = '';

            // Populate options for testbed types
            testbedDetails.subtypes.forEach(subtype => {
                const option = document.createElement('option');
                option.value = subtype.subtype;
                option.text = subtype.subtype;
                testbedTypeDropdown.appendChild(option);
            });
            // Trigger data sampling automatically if there's only one subtype
            if (testbedDetails.subtypes.length === 1) {
                console.log('Only one subtype found for the selected testbed. Sampling data automatically.');
                testbedTypeDropdown.value = testbedDetails.subtypes[0].subtype;
                populateSampledDataOptions(); // Trigger data sampling
            }
        } else {
            console.error('No subtypes found for the selected testbed.');
            alert('No subtypes available for the selected testbed.');
        }
    }
    window.populateTestbedOptions = populateTestbedOptions;

    // Fetch sampled data and display it
    async function populateSampledDataOptions() {
        const testbed = document.getElementById('testbed').value;
        const testbedType = document.getElementById('testbed-type').value;

        
        if (!testbed || !testbedType) {
            alert('Please select a testbed and testbed type.');
        }
        const testbedDetails = Object.values(testbeds).flat().find(t => t.testbed_name === testbed);
        const subtypeDetails = testbedDetails.subtypes.find(s => s.subtype === testbedType);
        document.getElementById('file-path').value = subtypeDetails.file_path;    
    
        // url compatible file path
        const filePath = encodeURIComponent(subtypeDetails.file_path);
        // endpoint is backend/data/mage/sample
        const response = await fetch(`/data/mage/sample?file_path=${filePath}`);
        const data = await response.json();
        document.getElementById('sampled-text').value = data.sampled_data.text;
        resizeTextarea(document.getElementById('sampled-text')); // enlarge to show the full text
        document.getElementById('correct-label').value = data.sampled_data.decision;
    }
    window.populateSampledDataOptions = populateSampledDataOptions;

    // Resize the textarea dynamically
    function resizeTextarea(textarea) {
        textarea.style.height = "auto";
        textarea.style.height = textarea.scrollHeight + "px";
    }

    // Show progress bar
    function showProgressBar() {
        const progressContainer = document.getElementById("progress-container");
        const progressBar = document.getElementById("progress-bar");
        progressContainer.style.display = "block";
        progressBar.style.width = "0%";
        progressBar.textContent = "0%";
    }

    // Update progress bar
    function updateProgressBar(progress) {
        const progressBar = document.getElementById("progress-bar");
        progressBar.style.width = progress + "%";
        progressBar.textContent = progress + "%";
    }

    // Hide progress bar
    function hideProgressBar() {
        const progressContainer = document.getElementById("progress-container");
        progressContainer.style.display = "none";
    }

    // Run inference and display model outputs
    async function runInference() {
        document.getElementById("longformer-prediction").textContent = "";
        document.getElementById("longformer-confidence").textContent = "";
        document.getElementById("bert-prediction").textContent = "";
        document.getElementById("bert-confidence").textContent = "";
        document.getElementById("watermark-prediction").textContent = "";
        document.getElementById("watermark-confidence").textContent = "";
        document.getElementById("num-tokens-scored").textContent = "";
        document.getElementById("num-green-tokens").textContent = "";
        document.getElementById("green-fraction").textContent = "";
        document.getElementById("z-score").textContent = "";
        document.getElementById("p-value").textContent = "";
        document.getElementById("detection-threshold").textContent = "";
        document.getElementById("watermarked-text").value = "";
        document.getElementById("macdet-prediction").textContent = "";
        document.getElementById("macdet-confidence").textContent = "";


        const inputMode = document.getElementById("input-mode").value;
        let text;

        if (inputMode === "manual") {
            text = document.getElementById("manual-text").value;
            if (!text) {
                alert("Please enter text.");
                return;
            }
        } else {
            text = document.getElementById("sampled-text").value;
            if (!text) {
                alert("Please select a testbed and testbed type to sample text.");
                return;
            }
        }
        var data = null;
        try {
            console.log("Running inference with text:", text);
            const response = await fetch("/inferAll", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text }),
            });
        
            if (!response.ok) throw new Error("Failed to fetch inference results.");
        
            data = await response.json();
            // Update DOM elements as before
        } catch (error) {
            console.error("Inference failed:", error);
            alert("An error occurred while fetching inference results. Please try again.");
        }
        console.log("Inference results:", data);

        if (data) {
            if (data.longformer) {
                document.getElementById("longformer-prediction").textContent = data.longformer.label;
                document.getElementById("longformer-confidence").textContent = `${(data.longformer.confidence * 100).toFixed(2)}%`; // Confidence in percentage
            } else {
                console.error("Received erroneous longformer data:", data.longformer);
                // alert("An error occurred while processing the longformer inference results. Please try again.");
            }

            if (data.finetuned) {
                document.getElementById("bert-prediction").textContent = data.finetuned.label;
                document.getElementById("bert-confidence").textContent = `${(data.finetuned.confidence * 100).toFixed(2)}%`; // Confidence in percentage
            } else {
                console.error("Received erroneous finetuned data:", data.finetuned);
                // alert("An error occurred while processing the finetuned inference results. Please try again.");
            }

            if (data.watermark) {
                // Populate the HTML elements with backend data
                document.getElementById("watermark-prediction").textContent = data.watermark.label || "Unknown";
                document.getElementById("watermark-confidence").textContent = `${(data.watermark.confidence * 100).toFixed(2)}%`; // Confidence in percentage
                document.getElementById("num-tokens-scored").textContent = data.watermark.num_tokens_scored || "N/A";
                document.getElementById("num-green-tokens").textContent = data.watermark.num_green_tokens || "N/A";
                document.getElementById("green-fraction").textContent = data.watermark.green_fraction.toFixed(4) || "N/A";
                document.getElementById("z-score").textContent = data.watermark.z_score.toFixed(4) || "N/A";
                document.getElementById("p-value").textContent = data.watermark.p_value.toExponential(6) || "N/A"; // Scientific notation
                document.getElementById("detection-threshold").textContent = data.watermark.detection_threshold || "N/A";
                
                // Populate the watermarked text area
                document.getElementById("watermarked-text").value = data.watermark.text || "No watermarked text available.";
                resizeTextarea(document.getElementById("watermarked-text"));
            } else {
                console.error("Received erroneous watermark data:", data.watermark);
                // alert("An error occurred while processing the watermark inference results. Please try again.");
            }
            
            if (data.macdet) {
                document.getElementById("macdet-prediction").textContent = data.macdet.label;
                document.getElementById("macdet-confidence").textContent = `${(data.macdet.confidence * 100).toFixed(2)}%`; // Confidence in percentage
            } else {
                console.error("Received erroneous macdet data:", data.macdet);
                // alert("An error occurred while processing the macdet inference results. Please try again.");
            }

        } else {
            console.error("Received erroneous data:", data);
            alert("An error occurred while processing the inference results. Please try again.");
        }
    }
    window.runInference = runInference;

    // Run inference with progress bar
    async function runInferenceWithProgressBar() {
        showProgressBar();
        document.getElementById("longformer-prediction").textContent = "";
        document.getElementById("longformer-confidence").textContent = "";
        document.getElementById("bert-prediction").textContent = "";
        document.getElementById("bert-confidence").textContent = "";
        document.getElementById("watermark-prediction").textContent = "";
        document.getElementById("watermark-confidence").textContent = "";
        document.getElementById("watermarked-text").value = "";

        const inferencePromise = runInference();
        let progress = 0;
        const minDuration = 3000;
        const startTime = Date.now();

        while (progress < 100) {
            const elapsedTime = Date.now() - startTime;
            progress = Math.min((elapsedTime / minDuration) * 100, 100);
            updateProgressBar(Math.round(progress));

            if (await Promise.race([inferencePromise, new Promise((resolve) => setTimeout(resolve, 100))]) === true) {
                updateProgressBar(100);
                break;
            }
            await new Promise((resolve) => setTimeout(resolve, 50));
        }
        hideProgressBar();
    }
    window.runInferenceWithProgressBar = runInferenceWithProgressBar;
});