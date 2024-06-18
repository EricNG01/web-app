import axios from 'axios'
import React, { useState } from 'react';

// Importing CSS
import './SearchBar.css'; 
import './App.css';
import './Response.css';

// Importing multimedia content
import microphoneIcon from './multimedia/microphone.svg'
import searchIcon from './multimedia/send.svg'
import redoIcon from './multimedia/redo.svg'
import nullImg from './multimedia/null.jpg'
import loader from './multimedia/loader.gif'

function App() {
  const [enquiry, setEnquiry] = useState("")
  const [encodedImage, setEncodedImage] = useState([])
  
  const startListening = () => {
    const recognition = new window.webkitSpeechRecognition() || new window.web;

    recognition.onresult = (event) => {
      // Get the transcripts 
      const text = event.results[0][0].transcript;
      setEnquiry(text);
    };

    // Stop when it is not getting any voice input for a while (1-2s)
    recognition.onend = () => {
      recognition.stop();
    };
    // Starting recording
    recognition.start();
  }
  const regen = async (event) => {
    const parentDiv = event.currentTarget.closest('.response'); // Finds the closest parent with class 'response'
    event.preventDefault()
    try {
      const imageToUpdate = parentDiv.querySelector('.image-response');
      imageToUpdate.src = loader;
      await axios.post('http://localhost:9527/imageGeneration', {
        enquiry: enquiry,
        instruction: parentDiv.querySelector('div').textContent
      }, {
        headers: {
          'Content-Type': 'application/json;charset=UTF-8'
        }
      }).then(res => {
        console.log(res.data.img)
        imageToUpdate.src = `data:image/png;base64,${res.data.img}`;
      })
    } catch (error) {
      console.error('Error fetching new image:', error);
    }  
  }
  const tts = (event) => {
    const parentDiv = event.currentTarget.closest('.response');
    const text = parentDiv.querySelector('div').textContent
    const synth = window.speechSynthesis;
    if (synth.speaking) {
      synth.cancel(); // Cancel the current speech
    }
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = 0.7;
    utterance.pitch = 1.5;
    setTimeout(1000);
    synth.speak(utterance);
  }
  const sendRequest = async (event) => {
    event.preventDefault()
    try {
      // language model API prompt
      await axios.post('http://localhost:9527/llm', {
        enquiry: enquiry,
      }, {
        headers: {
          'Content-Type': 'application/json;charset=UTF-8'
        }
      }).then(async (res) =>{
        console.log(res)
        if (res.status === 200) {

          console.log("success")
          console.log(res.data)

          var instruction = res.data.split("\n")
          var result = []
          for (var i = 0; i < res.data.split("\n").length - 1; i++) {
            console.log(instruction[i].slice(3, -1))
            // Image generation prompt
            await axios.post('http://localhost:9527/imageSearching', {
              enquiry: instruction[i].slice(3, -1),
            }, {
              headers: {
                'Content-Type': 'application/json;charset=UTF-8'
              }
            }).then(res => {
              console.log(res.data.img)
              result.push(
                <div className='response' id={`${i+1}`}>
                  <button 
                    className='instruction-response-button'
                    type="button" 
                    onClick={(e) => tts(e)}
                  >
                    <div>{instruction[i].slice(3, -1)}</div>
                  </button>
                  {res.data.img === null?
                    <img draggable={true} className='image-response' id={`image-response-${i+1}`} src={nullImg} alt="image-response" />
                  : <img draggable={true} className='image-response' id={`image-response-${i+1}`} src={`data:image/png;base64,${res.data.img}`} alt="image-response" />}
                  <button 
                    className="response-button" 
                    id={`${i+1}`}
                    type="button" 
                    onClick={e => regen(e)}>
                    <img src={redoIcon} alt="Redo" />
                  </button>
                </div>
              )
            })
          }
          setEncodedImage(result)
        } else {
          console.log(res.data.error)
        }
      })
    } catch (error) {
      console.log(error)
    }
    
  }

  return (
    <div>
      <header className="parent" style={{ minHeight: "10vh", minWidth:"100vw" }}>

        {/* Searching bar */}
        <div className="search-container">

          <div className='input-container'>

            <input 
              className="search-input"
              type='text'
              placeholder='How can I help you?'
              value={enquiry}
              onChange={(e) => setEnquiry(e.target.value)}
            />

            <button 
              className="icon-button" 
              id='icon-button-mic'
              type="button"
              onClick={startListening}
            >
              <img src={microphoneIcon} alt="Mic" />
            </button>

            <button 
              className="icon-button" 
              type="button" 
              id='icon-button-search'
              onClick={sendRequest}>
              <img src={searchIcon} alt="Search" />
            </button>

          </div>

        </div>

      </header>

      {/* Response from the models */}
      <div className="parent"  style={{ minHeight: "90vh", minWidth:"100vw" }}>
        {encodedImage}
      </div>
      
    </div>
  );
}

export default App;
