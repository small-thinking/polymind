# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.61] - 2024-08-06

### Added
- **Image Processing Utilities**: Added new utility functions to `polymind.core.utils`:
  - `encode_image_to_base64()`: Converts local image files to base64 strings for API calls
  - `is_valid_image_url()`: Validates image URLs for common image hosting services
- **Image Understanding Tool**: Added `ImageUnderstandingTool` to the media-gen example module:
  - Uses OpenAI's GPT-4o-mini API for image analysis
  - Supports both local image files and image URLs
  - Configurable prompts with default values
  - Optional JSON response format for structured output
  - Comprehensive error handling and metadata tracking

### Changed
- Updated version to 0.0.61
- Enhanced media-gen example with integration tests and documentation

### Technical Details
- Added type annotations and comprehensive error handling for image utilities
- Implemented automatic base64 encoding for local images
- Added support for multiple image formats and hosting services
- Created separate integration test structure for real API testing

## [0.0.60] - Previous version

### Added
- Initial release with core Polymind framework
- Multi-agent collaboration capabilities
- Task management system
- Tool integration framework
- Knowledge retrieval and indexing tools 