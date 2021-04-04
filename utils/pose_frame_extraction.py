
import cv2


def processing(movie_path):
    
    cap = cv2.VideoCapture(movie_path)
    nFrame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    process_count = 0
    
    while nFrame > process_count:
        
        # //////////////////////////////////
        #
        #      Write your code here.
        #
        # //////////////////////////////////
        
        process_count += 1
        print('\r> Count: %5d/%d' %(process_count, nFrame), end='')
    
    print('\n')
        

if __name__ == '__main__':
    
    movie_path = './carrying.mp4'
    processing(movie_path)

