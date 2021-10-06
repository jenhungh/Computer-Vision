# Import necessary functions
import numpy as np
import cv2
import skimage
import skimage.io
from loadVid import loadVid
import imageio
from opts import get_opts
from ar_helper import composeWarpedImg

opts = get_opts()


def main():
    # Load in necessary data
    src_frames = loadVid('../data/ar_source.mov')
    book_frames = loadVid('../data/book.mov')
    cv_cover = skimage.io.imread('../data/cv_cover.jpg')

    # Cut the zero padded region and convert to RGB
    src_frames = src_frames[:, 48:-48, :, ::-1]
    book_frames = book_frames[:, :, :, ::-1]
    size = len(src_frames)

    # Initialize composite frames
    composite_frames = []

    # Warp each frame  
    for index in range(len(book_frames)):
        # Pad at the end of src_frames with the beginning of src_frames until number of srcframes equals number of book_frames
        if index >= len(src_frames):
            append_frame = src_frames[index % size].reshape(1, src_frames.shape[1], src_frames.shape[2], src_frames.shape[3])
            src_frames = np.append(src_frames, append_frame, axis = 0)
        
        # Crop src_frames so that it has the same aspect ratio as the cv_cover
        src_frame = cv2.resize(src_frames[index], dsize = (cv_cover.shape[1], cv_cover.shape[0]))

        # Change the ratio at 435th frame to make sure that there are enough matching points
        if index == 435:
            opts.ratio = 0.8
        else:
            opts.ratio = 0.7
        
        # Use ar_helper.composeWarpedImg to place each src_frame correctly over each book_frame, using cv_cover as a reference for warping
        composite_frame = composeWarpedImg(cv_cover, book_frames[index], src_frame, opts)
        # Check composite frame
        # cv2.imshow('composite_frame', composite_frame)
        # cv2.waitKey(1)

        # Update composite frames
        composite_frames.append(composite_frame)
        print(f"Index {index} done.")

    # save composite frames into a video, where composite_frames is a list of image arrays of shape (h, w, 3) obtained above
    imageio.mimwrite('../result/ar.avi', composite_frames, fps=30)


if __name__ == '__main__':
    main()
