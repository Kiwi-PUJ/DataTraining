<h1 align="center"> Data Training </h1>
<h5 align="center"> A project to train CNNs for image segmentation </h5>

</p>
<p align="center">
<img src ="./documentation/media/log.png" alt="Logo" width="1200"/>
</p>

Considering the impact on the world of artificial intelligence and process automation in recent years, semi-autonomous systems have been developed that act responding to signals from their environment. An example is food delivery robots that establish their trajectory from images of their surroundings. To develop a system like this it is necessary to make use of segmentation algorithms trained to recognize obstacles, and to train these algorithms, databases of segmented images corresponding to the environment in which the robot will operate are required.

To train the selected models with the segmented dataset in [DataLabelling](https://github.com/Kiwi-PUJ/DataLabelling), the code available in this repository was developed.

**Status**: In development (code updates may be submitted after publication)

---
<h3 align="left"> Clone the data prediction repository </h3>

For the installation is necessary to clone the current repository in a local repository. If you don't have GitHub installed yet, you should.
If you are on Linux you can run the following command:

	sudo apt-get install git

Now you need to configure GitHub. For this run the following commands:

	git config --global user.name "user_name"
	git config --global user.email "email_id"

Otherwise, if you are not working on Linux, you can install GitHub from the [official website](https://desktop.github.com/). 

After installing and configuring GitHub on your computer, you must clone the repository. If you are on Linux, from the terminal you must access the path where you want the repository to be cloned and execute the following command:

	git clone https://github.com/Kiwi-PUJ/DataLabelling.git

*If you are not working on Linux you must clone the repository from the GitHub application.*

**Now the application files will be on your device.**

---
<h3 align="left"> Build and Run Docker Image </h3>

The Image Labelling App dependencies, compilation, and configuration are packaged in a Docker Image. Before continuing, make sure you have Docker installed on your device. If it is not installed and you are working on Linux, you can run the following commands in a terminal:

	sudo apt-get update
 	sudo apt-get install docker-ce docker-ce-cli containerd.io


This can take a while. When the installation is done you can test it by running the Docker image *hello-world*. 

	sudo docker run hello-world


If the installation was done correctly, you should see some informational text on the screen.

Besides this, there are other post-installation steps for Linux. To complete them, run the following commands:

	sudo groupadd docker
	sudo usermod -aG docker $USER
	newgrp docker


*If you are not working on Linux, you can install by visiting the* [official website](https://docs.docker.com/get-docker/).

To run the data prediction Docker image, verify that you are on the **DataTraining** path and run the following command:

	bash start.sh

If it's the first time probably is going to take a while.

If all goes well, you should be seeing a progress bar for each of the training proccess and the trained models should be saving to the **files** folder.

---
<h2 align="left"> Code Documentation </h2>

The code documentation was done using PEP8 and Doxygen and is available [here](http://trainingkiwipuj.125mb.com/)

---
<h2 align="left"> This project is being developed by: </h2>

✈️ Andrea Juliana Ruiz Gómez, [GitHub](https://github.com/andrearuizg), Email: andrea_ruiz@javeriana.edu.co

🏎️ Pedro Elí Ruiz Zárate, [GitHub](https://github.com/PedroRuizCode), Email: pedro.ruiz@javeriana.edu.co


<h3 align="left"> With the support of: </h3>

👨🏻‍🏫 Francisco Carlos Calderón Bocanegra, [GitHub](https://github.com/calderonf)

👨🏻‍💻 John Alberto Betancourt Gonzalez, [GitHub](https://github.com/JohnBetaCode)
