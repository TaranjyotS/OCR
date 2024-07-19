import cv2
import numpy as np
import os
import re

MY_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(MY_DIR, 'images')

def knn():
    """Finds the test data's closest match in the feature space.
    
    Returns:
        knn: Vector containing each input sample's prediction (classification or regression) results.
    """
    digits_image = cv2.imread(os.path.abspath(os.path.join(IMAGES_DIR, 'training_datasets/digits.jpg')))
    gray = cv2.cvtColor(digits_image, cv2.COLOR_BGR2GRAY)
    small = cv2.pyrDown(digits_image)
    cv2.imshow('Digits Image', small)

    cv2.waitKey(500)
    cv2.destroyAllWindows()

    # Split the image to 5000 cells, each cell 20 x 20
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # Convert list data type to numpy array of shape (50,100,20,20)
    x = np.array(cells)
    print('The shape of our cells array is '+str(x.shape))

    # Split the full data_set into two segments
    train = x[:, :70].reshape(-1, 400).astype(np.float32)  # size = 3500x400
    test = x[:, 70:100].reshape(-1, 400).astype(np.float32)  # size = 1500x400

    # Create label for train and test data
    k = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_labels = np.repeat(k, 350)[:, np.newaxis]
    test_labels = np.repeat(k, 150)[:, np.newaxis]

    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    _, results, neighbours, dist = knn.findNearest(test, k=3)

    print ('Results: ', results,'\n')
    print ('Neighbours: ', neighbours,'\n')
    print ('Distances: ', dist)

    matches = results == test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct * (100.0 / results.size)
    print('Accuracy is = %.2f' % accuracy + '%')

    return knn


def x_cord_contour(contour):
    """Gets the weighted average (moment) of the image pixels' intensities.

    Args:
        contour: Detected contour of the image.

    Returns:
        A dictionary of all moment values calculated.
    """
    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        return int(M['m10']/M['m00'])
    else:
        return int(0)


def makeSquare(dimensions):
    """Creates a square around the identified character.
    
    Args:
        dimensions: Dimesions of the shape.

    Returns:
        doublesize_square: Dimensions of a square.
    """
    black = [0, 0, 0]
    img_dim = dimensions.shape
    height = img_dim[0]
    width = img_dim[1]

    if height == width:
        square = dimensions
        return square
    else:
        doublesize = cv2.resize(dimensions, (2*width, 2*height), interpolation=cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2

        if height > width:
            pad = int((height - width)/2)
            doublesize_square = cv2.copyMakeBorder(doublesize, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=black)

        else:
            pad = int((width - height)/2)
            doublesize_square = cv2.copyMakeBorder(doublesize, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=black)

    # doublesize_square_dim = doublesize_square.shape

    return doublesize_square


def resize_to_pixel(dimensions, image):
    """Resizes the image to pixel according to the given dimensions.
    
    Args:
        dimensions: Dimesions of the image.
        image: Image that is to be resized to pixels.

    Returns:
        ReSizedImg: Resized image.
    """
    buffer_pix = 4
    dimensions = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0]*r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    black = [0, 0, 0]
    if height_r > width_r:
        resized = cv2.copyMakeBorder(resized, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=black)
    if height_r > width_r:
        resized = cv2.copyMakeBorder(resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value=black)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized, p, p, p, p, cv2.BORDER_CONSTANT, value=black)

    # img_dim = ReSizedImg.shape
    # height = img_dim[0]
    # width = img_dim[1]

    return ReSizedImg

def ocr(knn, number_image):
    """Performs data extraction from the given image to recognise the characters.

    Args:
        knn: 
        number_image: The character image for whih wqe want to perform the OCR.
    """
    # Convert to gray image
    gray = cv2.cvtColor(number_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Image', number_image)
    cv2.imshow('Gray', gray)
    cv2.waitKey(200)

    # Convert to blurred image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('Blurred', blurred)
    cv2.waitKey(200)

    # Detect edges of the image
    edged = cv2.Canny(blurred, 30, 150)
    cv2.imshow('Edged', edged)
    cv2.waitKey(200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=x_cord_contour, reverse=False)


    full_number = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        cv2.drawContours(number_image, contours, -1, (0, 255, 0), 3)
        cv2.imshow('Contours', number_image)

        if w >= 5 and h >= 25:
            roi = blurred[y:y + h, x:x + w]

            _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
            squared = makeSquare(roi)
            final = resize_to_pixel(20, squared)
            cv2.imshow('Final', final)
            final_array = final.reshape((1, 400))
            final_array = final_array.astype(np.float32)
            _, result, _, _ = knn.findNearest(final_array, k=1)
            number = str(int(result[0]))
            full_number.append(number)


            # Identification of the character
            cv2.rectangle(number_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(number_image, number, (x, y+155), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
            cv2.imshow('Image', number_image)
            cv2.waitKey(200)

    cv2.destroyAllWindows()
    print(f'The numbers identified are : {full_number}')


def do_it():
    """Script entry point."""
    knn_value = knn()
    for image in os.listdir(os.path.join(IMAGES_DIR, 'sample_datasets')):
        if re.match('num[0-9]', image) and image.endswith('.png'):
            number_image = cv2.imread(os.path.abspath(os.path.join(IMAGES_DIR, 'sample_datasets', image)))
            ocr(knn_value, number_image)


if __name__ == '__main__':
    do_it()
