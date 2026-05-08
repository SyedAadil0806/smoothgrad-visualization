# Setup Instructions

## 1. Create Virtual Environment

```bash
python -m venv venv
```

---

## 2. Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3. Install Required Libraries

```bash
pip install torch torchvision matplotlib pillow numpy
```

---

# How to Run

## 1. Add Images

Place your test images inside the `test_images` folder.

Example:

test_images/
├── cat.jpg
├── dog.jpg


---

## 2. Run the Project

```bash
python main.py
```

---

## 3. Select Image

The terminal will display all images found in the `test_images` folder.

Enter the number corresponding to the image you want to process.

Example:

```text
1. cat.jpg
2. dog.jpg

Enter the image number you want to use:
```

---

## 4. Output

The generated output image will be shown to you as well as automatically be saved inside the `outputs` folder.

## 5. GitHub link

https://github.com/SyedAadil0806/smoothgrad-visualization