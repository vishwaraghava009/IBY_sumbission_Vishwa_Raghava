# Docker Usage

## Building the image

*This is assuming you have docker and docker compose installed and running.*

1. Open a terminal and set your current directory to the "docker" directory in your clone of this repo.
2. Adjust the "docker-compose.yml" file to match your needs. Please see the in file comments for additional information on how to prepare your file.
3. Once ready, run the command "docker-compose build" to build the "cohee1207/sillytavern-extras" docker image.

If you are running the docker in an environment where you do not utilize the docker-compose.yml file (IE: Windows Docker or Unraid), then you will need to specify enabled modules as an "Extra Parameter". Any flags, including the required "--enable-modules=" flag, will need to be put there.

