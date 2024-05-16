// import { useState } from "react";
// import reactLogo from "./assets/react.svg";
// import viteLogo from "/vite.svg";
import { useState } from "react";
import "./App.css";
import BackgroundImage from "./components/BackgroundImage";
import ImageForm from "./components/ImageForm";
import PredictionResult from "./components/PredictionResult";
import Title from "./components/Title";
import Subtitle from "./components/Subtitle";

function App() {
  // const [count, setCount] = useState(0);
  const [prediction, setPrediction] = useState("");
  const [displayResult, setDisplayResult] = useState(false);

  const predictionLabels = {
    0: "abstract",
    1: "genre painting",
    2: "landscape",
    3: "portrait",
  };
  function handlePrediction(predictionNumber) {
    setPrediction(predictionLabels[predictionNumber]);
  }

  return (
    <div className="flex flex-col items-center justify-center">
      <BackgroundImage>
        <div className="mt-16 w-full h-3/5 flex flex-col justify-stretch items-center gap-12">
          <Title />
          <Subtitle />
          <ImageForm
            handlePrediction={handlePrediction}
            setDisplayResult={setDisplayResult}
          />
          {displayResult && <PredictionResult prediction={prediction} />}
        </div>
      </BackgroundImage>
    </div>
  );
}

export default App;
