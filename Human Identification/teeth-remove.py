import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
image_path = 'C:/Users/Md-Ferdous/Desktop/dental x-ray images/26.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image_rgb.shape[:2]

mask = np.zeros((height, width), dtype=np.uint8)

class MultiToothSelector:
    def __init__(self, ax, image):
        self.canvas = ax.figure.canvas
        self.image = image
        self.ax = ax
        self.lasso = LassoSelector(ax, onselect=self.on_select)
        self.count = 0
        self.ax.set_title("Draw around teeth to remove. Press Enter when done.")

       
        self.cid = self.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_select(self, verts):
        global mask
        path = Path(verts)
        y, x = np.mgrid[:height, :width]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        mask_flat = path.contains_points(coors)
        mask_points = mask_flat.reshape((height, width))
        
        
        mask[mask_points] = 255
        
       
        self.ax.imshow(np.ma.masked_where(mask == 0, mask), cmap='jet', alpha=0.5)
        self.canvas.draw_idle()
        self.count += 1
        print(f"[INFO] Tooth {self.count} selected. Draw next or press Enter to finish.")

    def on_key_press(self, event):
        if event.key == 'enter':
            print("[INFO] Selection finished.")
            self.lasso.disconnect_events()
            self.canvas.mpl_disconnect(self.cid)
            plt.close(self.canvas.figure) 

# Display image for multiple teeth selection
fig, ax = plt.subplots()
ax.imshow(image_rgb)
selector = MultiToothSelector(ax, image_rgb)
plt.show()

# Inpaint all selected regions
inpainted = cv2.inpaint(image, mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)

# Show results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original with Mask")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("After Tooth Removal")
plt.imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.tight_layout()
plt.show()

# Save the result
cv2.imwrite("tooth_removed_multiple.jpg", inpainted)