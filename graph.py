import pydot

# Load the DOT file
graph = pydot.graph_from_dot_file("model.dot")

# Write the DOT graph to a PNG image
graph[0].write_png("output_image.png")