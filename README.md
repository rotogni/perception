# run code
compile main.cpp using bash:
    clang++ main.cpp pose_estimation.cpp visualization.cpp -o main -std=c++17 \
    -I/opt/homebrew/Cellar/opencv/4.11.0_1/include/opencv4 \
    -L/opt/homebrew/Cellar/opencv/4.11.0_1/lib \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs \
    -lopencv_features2d -lopencv_calib3d -lopencv_viz
run:
./main
# credits
kitti Dataset :
  Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun (2013) Vision meets Robotics: The KITTI Dataset, International Journal of Robotics Research (IJRR)

Morocco Dataset :
Meyer, L., Smíšek, M., Fontan Villacampa, A., Oliva Maza, L., Medina, D., Schuster, M. J., Steidle, F., Vayugundla, M., Müller, M. G., Rebele, B., Wedler, A., & Triebel, R. (2021). The MADMAX data set for visual‐inertial rover navigation on Mars. Journal of Field Robotics, 1– 21. https://doi.org/10.1002/rob.22016