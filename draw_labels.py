def draw_labels(ax, fig, rect, text):
    # Get the landmarks/parts for the face in box d.
    # Draw the face landmarks on the screen.
    rectangle = Rectangle((rect.left(), rect.top()), rect.width(), rect.height(), linewidth = 1, edgecolor = 'b', facecolor = 'none')
    ax.add_patch(rectangle)
    ax.text(rect.center().x, rect.center().y, text, ha = "center", va = "center", color="#FFFFFF")
    
