# Face Mask Detection Project
![facemask](https://github.com/pratyakshsuri2003/Yolo-Projects/assets/115720372/9a62fa07-a8f6-4ab5-a006-22124420f14d)
A Face Mask Detection project that uses the Super Gradients library and Roboflow for building a detection model.
## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- Detects whether a person is wearing a face mask or not.
- Utilizes the Super Gradients library for training and evaluation.
- Uses Roboflow for data preprocessing.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- Pip (Python package manager)
- Super Gradients, Imutils, Roboflow, and other dependencies (see [requirements.txt](requirements.txt))

### Installation

1. Clone this repository:

   ```sh
   git clone https://github.com/yourusername/Face_Mask_Detection.git
   cd Face_Mask_Detection

## Usage
- Train your Face Mask Detection model using Super Gradients, Roboflow, and other components mentioned in your code.
- Use the trained model for inference or integrate it into your application.

  ```sh
  # Sample code for inference
  from super_gradients.training import models
  from super_gradients.training import Trainer
  
  # Load the trained model
  model = models.YourModel()
  model.load_weights('path/to/your/weights')
  
  # Perform inference on an image
  image = Image.open('path/to/your/image.jpg')
  detections = model.predict(image)

  # Process the detection results as needed

## Contributing
To contribute to this project, follow these steps:

- Fork this repository.
- Create a new branch: git checkout -b feature/new-feature.
- Make your changes and commit them: git commit -m 'Add new feature'.
- Push to the branch: git push origin feature/new-feature.
- Submit a pull request.

### License
Copyright (c) [2023] PRATYAKSH SURI]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
