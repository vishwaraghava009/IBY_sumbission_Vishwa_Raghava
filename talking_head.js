function fetchFrames() {
    console.log('Fetching frames...');
    // Use a static image URL for testing
    const staticImageUrl = '/home/iby_vishwa/Documents/SillyTavern-extras/talkinghead.png';  // Replace with a valid static image URL

    fetch(staticImageUrl)
        .then(response => response.blob())
        .then(blob => {
            console.log('Blob fetched:', blob);
            console.log('Blob size:', blob.size);
            console.log('Blob type:', blob.type);

            if (blob.size > 0) {
                const img = new Image();
                img.onload = function () {
                    console.log('Image loaded, drawing on canvas...');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0);
                };
                img.onerror = function () {
                    console.error('Error loading image from blob');
                };
                const objectURL = URL.createObjectURL(blob);
                console.log('Object URL created:', objectURL);
                img.src = objectURL;
                console.log('Image source set to object URL:', img.src);
            } else {
                console.error('Blob size is 0');
            }
        })
        .catch(error => {
            console.error('Error fetching static image:', error);
        });
}
