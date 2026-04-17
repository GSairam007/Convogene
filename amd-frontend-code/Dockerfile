FROM node:20-alpine as builder
 
# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json to the working directoryCOPY package.json .
COPY package.json .
COPY package-lock.json .

# Install dependencies
RUN npm install --force

# Copy the remaining application code to the working directory
COPY . .

# Build the React app
RUN npm run build

# Expose port 3000 to the outside world
EXPOSE 3000

# Command to run the React app
CMD ["npm", "start"]
