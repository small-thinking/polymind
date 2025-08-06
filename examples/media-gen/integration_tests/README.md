# Integration Tests

This folder contains integration tests that require real API calls and external resources.

## Image Understanding Integration Test

### Prerequisites
- OpenAI API key in `.env` file
- Internet connection
- Test image file (`test_image.png`)

### Running the Test

```bash
# From the media-gen directory
python integration_tests/test_image_understanding.py
```

### What it does
- Loads the test image (`test_image.png`)
- Generates an image generation prompt that could be used to recreate the image
- Calls OpenAI's GPT-4o-mini API
- Displays the generated prompt and metadata

### Expected Output
The test will show:
- ✅ Confirmation that test image and API key are found
- 📝 The prompt being used
- 📋 The generated image generation prompt
- 📊 Token usage metadata

### Notes
- This test is not run automatically with unit tests
- It requires a valid OpenAI API key
- It makes real API calls and may incur costs
- The test image should be placed in this folder 