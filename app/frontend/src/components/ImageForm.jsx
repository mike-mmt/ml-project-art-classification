/* eslint-disable react/prop-types */
// eslint-disable-next-line no-unused-vars
import React from "react";
// import { useResizeImage } from "react-image";
import { useState } from "react";
import axios from "axios";
import { Button } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";

export default function ImageForm({ handlePrediction, setDisplayResult }) {
  // const [selectedFile, setSelectedFile] = useState(null);
  //   const [processedImage, setProcessedImage] = useState(null);
  const [previewFile, setPreviewFile] = useState("");
  const [isFileSelected, setIsFileSelected] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleChange = (event) => {
    setDisplayResult(false);
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreviewFile(URL.createObjectURL(file));
    setIsFileSelected(true);
  };

  const handleUpload = async () => {
    // console.log(import.meta.env.VITE_BACKEND_URL + "/upload-image");
    const reader = new FileReader();
    reader.readAsDataURL(selectedFile); // Use processedImage if available

    reader.onload = async (event) => {
      const base64Image = event.target.result;
      setDisplayResult(true);
      try {
        const response = await axios.post(
          import.meta.env.VITE_BACKEND_URL + "/upload-image",
          {
            image: base64Image,
          }
        );
        handlePrediction(response.data.prediction);
        console.log("Image uploaded successfully:", response.data);
      } catch (error) {
        // Handle upload error
        console.error("Upload error:", error);
      }
    };
  };
  return (
    <div className="flex flex-row justify-center items-center w-full fade-in-slow anim-delay-150">
      <Button
        sx={{ backgroundColor: "#6c34ba", margin: "0px" }}
        component="label"
        role={undefined}
        variant="contained"
        tabIndex={-1}
        startIcon={<CloudUploadIcon />}
      >
        Upload file
        <input
          style={{ display: "none" }}
          type="file"
          accept="image/*"
          onChange={handleChange}
        />
      </Button>
      {previewFile && (
        <img src={previewFile} alt="Preview" className="ml-8 h-[256px]" />
      )}
      <button
        className="ml-8"
        disabled={!isFileSelected}
        type="button"
        onClick={handleUpload}
      >
        Do the magic
      </button>
    </div>
  );
}
