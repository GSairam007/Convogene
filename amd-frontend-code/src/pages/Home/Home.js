import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { styled, useTheme } from "@mui/material/styles";
import RandomQueries from "../../components/IconsHolder/RandomQueries/RandomQueries";
import CustomChatbotComponent from "../Mini_Chatbot/CustomChatbotComponent";
import AddIcon from "@mui/icons-material/Add";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import micActiveLogo from "../../assets/Images/stop-button.png";
import thunbup from "../../assets/svg/g1.svg";
import thundown from "../../assets/svg/g1-1.svg";
import { ThumbUpAlt, ThumbDownAlt } from "@mui/icons-material";
import remarkGfm from "remark-gfm";
import CachedIcon from "@mui/icons-material/Cached";
import Shimmer from "./Shimmer"; // Import the Shimmer component
import {
  IconButton,
  Drawer,
  Divider,
  Button,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  RadioGroup,
  FormControl,
  FormControlLabel,
  Radio,
  Box,
  Modal,
} from "@mui/material";

import Typography from "@mui/material/Typography";

import {
  Menu as MenuIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
  ScreenSearchDesktop as ScreenSearchDesktopIcon,
  Send as SendIcon,
} from "@mui/icons-material";
import axios from "axios";

import Markdown from "react-markdown";

import infobellImg from "../../assets/Images/infobellLogo.png";
import newchat from "../../assets/Images/icons8-chat-64.png";
import openaiimg from "../../assets/Images/openai-logo-0.png";
import styles from "./Home.module.css";

import KeyboardVoiceIcon from "@mui/icons-material/KeyboardVoice";

import WidgetsOutlinedIcon from "@mui/icons-material/WidgetsOutlined";

import Collapse from "@mui/material/Collapse";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import Tooltip from "@mui/material/Tooltip";
import CommentIcon from "@mui/icons-material/Comment";
import ThumbDownOffAltIcon from "@mui/icons-material/ThumbDownOffAlt";
import ThumbUpOffAltIcon from "@mui/icons-material/ThumbUpOffAlt";
import { common } from "@mui/material/colors";
const sdk = require("microsoft-cognitiveservices-speech-sdk");
// const fs = require('fs').promises;
const SPEECH_KEY = "f4a8f5be7801494fa47bc87d6d8ca31d";
const SPEECH_REGION = "eastus";
const speechConfig = sdk.SpeechConfig.fromSubscription(
  SPEECH_KEY,
  SPEECH_REGION
);
speechConfig.speechRecognitionLanguage = "en-US";
speechConfig.speechSynthesisVoiceName = "en-US-AvaMultilingualNeural";
//Key1 : 334e16bf5eb643a6b8ebd8f0e61bf937u0i
//Key2 : 8bcb5f3956be4afdbbbc3c3b937fa437
//Region : eastus
//Endpoint : https://eastus.api.cognitive.microsoft.com/

const Home = () => {
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(true);
  const [searchValue, setSearchValue] = useState("");
  const [messages, setMessages] = useState([]);
  console.log(messages, "values inside message ");
  const [answers, setAnswers] = useState("");

  const [widthM, setWidthM] = useState("100%");
  const [isChatbotOpen, setIsChatbotOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [selectedOption, setSelectedOption] = useState("jira");
  // const [isResponseComplete, setIsResponseComplete] = useState(false); // Track response completion
  const [relatedQuestions, setRelatedQuestions] = useState([]);
  const queryRef = useRef(null);

  // const { profile } = location.state || {};
  const [logout, setLogout] = useState(false);
  const [open, setOpen] = useState(true);
  const theme = useTheme();
  let [answerFlag1, setAnswerFlag1] = useState(true);
  let [answerFlag2, setAnswerFlag2] = useState(true);
  let [answerFlag3, setAnswerFlag3] = useState(true);
  var [selectedFile, setSelectedFile] = useState();

  const handleOpenChatbot = () => setIsChatbotOpen(true);
  const handleCloseChatbot = () => setIsChatbotOpen(false);

  const [document, setDocument] = useState(false);
  const [link, setLink] = useState(false);
  const [maxTokens, setMaxTokens] = useState(100); // Default value
  const [temperature, setTemperature] = useState(0.5);
  const [topk, setTopk] = useState(2);
  const depthRef = useRef(null);
  const linkref = useRef("");
  const llm_endpoint_ref = useRef("");
  const embeddingRef = useRef("");
  const indexref = useRef("");

  const dimensionsRef = useRef("");
  const metricRef = useState("");
  const [monitoring, setMonitoring] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [recognizer, setRecognizer] = useState(null);
  const [selectedItem, setSelectedItem] = useState("");

  const [isCollapse5, setIsCollapse5] = useState(false);
  const [loadingChatHistory, setLoadingChatHistory] = useState(false);
  const [isCustom, setIsCustom] = useState(false);
  const [index, setIndex] = useState(false);
  const [config, setConfig] = useState(false);
  const [llm_endpoint, setLlm_endpoint] = useState("");
  const [indexes, setIndexes] = useState();
  const [files, setFiles] = useState();
  const [selectedIndex, setSelectedIndex] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isNewChat, setIsNewChat] = useState(true);
  const [isLargeScreen, setIsLargeScreen] = useState(true);
  const [Historyfeed, setHistoryfeed] = useState(false);
  const [selectedChatHistory, setSelectedChatHistory] = useState("");
  const [selectedSentiment, setSelectedSentiment] = useState("All");
  const today = new Date();

  const date_for_file = today.toISOString().split("T")[0];

  const handleCollapse5 = async () => {
    setLoadingChatHistory(true);
    const response = await axios.get(
      "/list_files"
    );
    console.log(response.data);
    setFiles(response.data);
    setIsCollapse5(true);
    setLoadingChatHistory(false);
  };

  const transformData = (rawData) => {
    let transformedData = [];
    rawData.forEach((item) => {
      const userMessage = { answer: item.Query, sender: "user" };
      const botMessage = { sender: "bot", display: 1, answer: item.Response };
      transformedData.push(userMessage, botMessage);
    });
    return transformedData;
  };

  const get_file = async (eventOrFileName) => {
    let fileName;
  
    if (typeof eventOrFileName === "string") {
      // Called from handlehistoryfeed
      fileName = eventOrFileName;
    } else if (eventOrFileName?.target?.value) {
      // Called from files list click
      fileName = eventOrFileName.target.value;
    } else {
      console.error("Invalid file input");
      return;
    }
  
    console.log("Fetching file:", fileName);
  
    setSelectedFile(fileName);
    setSelectedIndex(fileName);
    setMessages([]);
  
    try {
      const response = await axios.post(
        "/one_file",
        { file: fileName }
      );
  
      console.log("File response:", response.data);
  
      const transformedMessages = transformData(response.data);
      setMessages(transformedMessages);
      setIsOpen(false);
      setHistoryfeed(true);
    } catch (error) {
      console.error("Error fetching file:", error);
    }
  };
  

  useEffect(() => {
    const audioConfig = sdk.AudioConfig.fromDefaultMicrophoneInput();
    const speechConfig = sdk.SpeechConfig.fromSubscription(
      "f4a8f5be7801494fa47bc87d6d8ca31d",
      "eastus"
    );

    // Create the recognizer
    const speechRecognizer = new sdk.SpeechRecognizer(
      speechConfig,
      audioConfig
    );
    setRecognizer(speechRecognizer);

    return () => {
      if (speechRecognizer) {
        speechRecognizer.close();
      }
    };
  }, []);

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [feedback, setFeedback] = useState({
    comment: "",
    like: "",
    Query: "",
    Response: "",
  });
  const [activeIcon, setActiveIcon] = useState({}); // Track the active icon

  console.log(feedback, "feedback value");
  const handleThumbClick = (reaction, index, value) => {
    console.log("Reaction:", reaction); // "yes" or "no"
    console.log("Question answered:", value); // The user question

    // Set the active state for the clicked icon and reset the other icon for the same index
    setActiveIcon((prev) => ({
      ...prev,
      [`yes_${index}`]: reaction === "yes", // Set active state for thumbs-up (yes) icon
      [`no_${index}`]: reaction === "no", // Set active state for thumbs-down (no) icon
    }));

    const like = reaction;
    console.log(`Thumb clicked with value: ${value}`);

    // Update feedback state with question and answer for API
    setFeedback((prev) => ({
      ...prev,
      like: like, // Store the reaction (like or dislike)
      Response: value.answer, // Store the bot's response
      Query: value.question, // Store the user's question
    }));

    // Directly pass the updated state to the API call
    feedbackApi({
      Query: value.question,
      Response: value.answer,
      Feedback: like === "yes" ? "Positive" : "Negative",
    });
  };

  const feedbackApi = async (formattedFeedback) => {
    try {
      const response = await axios.post(
        "/fetch_feedback",
        formattedFeedback
      );
      console.log("Response:", response.data);
    } catch (error) {
      console.error("Error occurred while sending feedback:", error);
    }
  };

  const handleCommentClick = (index, value) => {
    // Log the feedback details

    // Update the feedback state
    setFeedback((prev) => ({
      ...prev,
      Query: value.question, // Store the question
      Response: value.answer, // Store the bot's answer
    }));

    // Open the modal
    setIsModalOpen(true);
  };

  const handlefeedback = async (e) => {
    e.preventDefault();

    const payload = {
      Feedback_text: feedback.comment,
      Query: feedback.Query,
      Response: feedback.Response,
    };

    try {
      const response = await axios.post(
        "/fetch_feedback",
        payload
      );
      console.log("Response:", response.data);
      if (response.data.code === "200") {
        closeModal();
      }
      // Handle success logic here (e.g., display a success message)
    } catch (error) {
      console.error("Error submitting feedback:", error);

      // Handle error logic here (e.g., display an error message)
      if (error.response) {
        // Server responded with a status other than 2xx
        console.error("Response data:", error.response.data);
        console.error("Status code:", error.response.status);
      } else if (error.request) {
        // No response received from the server
        console.error("No response received:", error.request);
      } else {
        // Something happened while setting up the request
        console.error("Error setting up request:", error.message);
      }
    }
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  const handlehistoryfeed = async (value) => {
    setSelectedSentiment(value); // Update selected sentiment state
  
    console.log(value, " value for hisfeed");
  
    if (value === "All") {
      get_file(selectedFile); 
    } else {
      const payload = {
        sentiment: value,
        file: selectedFile,
      };
  
      setMessages([]);
  
      try {
        const response = await axios.post(
          "/filter_feedback",
          payload
        );
        console.log("Response:", response.data);
  
        const extractedEntries = transformData(response.data); // Extract the entries array safely
        console.log(extractedEntries, "transformedMessages");
  
        setMessages(extractedEntries);
      } catch (error) {
        // Handle error
        if (error.response) {
          console.error("Response data:", error.response.data);
          console.error("Status code:", error.response.status);
        } else if (error.request) {
          console.error("No response received:", error.request);
        } else {
          console.error("Error setting up request:", error.message);
        }
      }
    }
  };
  
  const correctSpecialWords = (text) => {
    return text
      .split(" ")
      .map((word) => {
        switch (word.toLowerCase()) {
          case "amd epic":
            return "AMD EPYC";
          case "amd risen":
            return "AMD RYZEN";
          case "processes":
            return "processors";
          case "epic":
          case "pick":
            return "EPYC";
          case "risen":
          case "horizon":
          case "rise and":
            return "RYZEN";
          case "amd":
          case "md":
          case "mda":
            return "AMD";
          default:
            return word;
        }
      })
      .join(" ");
  };

  const startListening = async () => {
    if (recognizer) {
      setIsListening(true);

      recognizer.startContinuousRecognitionAsync(
        () => {
          console.log("Continuous recognition started");
        },
        (error) => {
          console.error("Error starting continuous recognition:", error);
          setIsListening(false);
        }
      );

      let previousWords = [];
      let currentSearchValue = "";

      recognizer.recognizing = (s, e) => {
        if (e.result.reason === sdk.ResultReason.RecognizingSpeech) {
          const currentWords = e.result.text.split(" ");
          const newWords = currentWords.filter(
            (word) => !previousWords.includes(word)
          );

          if (newWords.length > 0) {
            const interimText = correctSpecialWords(newWords.join(" "));
            currentSearchValue = (
              currentSearchValue +
              " " +
              interimText
            ).trim();

            console.log("Interim words:", interimText);
            console.log("Updated Search Value:", currentSearchValue);

            setSearchValue(currentSearchValue);
            previousWords = [...previousWords, ...newWords];
          }
        }
      };

      recognizer.recognized = (s, e) => {
        if (e.result.reason === sdk.ResultReason.RecognizedSpeech) {
          setTranscription((prevTranscription) => {
            // Correct the recognized text
            const correctedText = correctSpecialWords(e.result.text);
            const newTranscription = prevTranscription + " " + correctedText;

            console.log("RECOGNIZED:", e.result.text);
            console.log("Corrected Text:", correctedText);
            console.log("New Transcription:", newTranscription);

            setSearchValue(newTranscription);
            return newTranscription;
          });
        } else if (e.result.reason === sdk.ResultReason.NoMatch) {
          console.log("NOMATCH: Speech could not be recognized.");
        }
      };
    }
  };

  const stopListening = () => {
    setTranscription("");
    if (recognizer) {
      recognizer.stopContinuousRecognitionAsync(
        () => {
          console.log("Continuous recognition stopped");
          setIsListening(false);
        },
        (error) => {
          console.error("Error stopping continuous recognition:", error);
        }
      );
    }
  };

  const handleRelatedQuestionClick = (question) => {
    getAnswer(question);
  };

  const DrawerHeader = styled("div")(({ theme }) => ({
    display: "flex",
    alignItems: "center",
    padding: theme.spacing(0, 1),

    ...theme.mixins.toolbar,
    justifyContent: "flex-end",
  }));

  const handleDrawerOpen = () => {
    setOpen(true);
  };

  const drawerWidth = 280;

  const handleInputChange = (event) => {
    setSearchValue(event.target.value);
  };

  const getAnswer = async (temp) => {
    setIsNewChat(false);
    const query = temp || searchValue.trim();
    if (!query) {
      return;
    }

    setSearchValue("");
    setTranscription("");
    stopListening();

    const userMessage = { answer: query, sender: "user" };
    const botMessage = {
      sender: "bot",
      display: 0,
      answer: "",
      isProcessing: true,
    };
    setMessages([...messages, userMessage, botMessage]);
    setIsOpen(false);

    if (query) {
      try {
        const response = await fetch(
          "/rag_qa_api_stream",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: query }),
          }
        );

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let responseText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          responseText += chunk;
          const currentResponseText = responseText;
          setMessages((prevMessages) => {
            const newMessages = [...prevMessages];
            newMessages[newMessages.length - 1].answer = currentResponseText;
            newMessages[newMessages.length - 1].isProcessing = false;
            return newMessages;
          });
        }

        const relatedQuestionsResponse = await fetch(
          "/related_questions",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ prompt: query, answer: responseText }),
          }
        );

        const relatedQuestionsData = await relatedQuestionsResponse.json();
        const newRelatedQuestions =
          relatedQuestionsData.related_questions || [];

        setMessages((prevMessages) => {
          const newMessages = [...prevMessages];
          newMessages.push({
            sender: "bot_rq",
            answer: JSON.stringify(newRelatedQuestions),
          });
          return newMessages;
        });

        console.log("Related Questions:", relatedQuestions); // Log related questions

        setTranscription("");
      } catch (error) {
        console.error("Error in llm:", error);
      }
    }
  };

  useEffect(() => {
    const handleKeyPress = (event) => {
      if (event.key === "Enter") {
        getAnswer();
      }
    };

    const inputElement = queryRef.current;
    if (inputElement) {
      inputElement.addEventListener("keydown", handleKeyPress);
    }

    return () => {
      if (inputElement) {
        inputElement.removeEventListener("keydown", handleKeyPress);
      }
    };
  });

  const monitor = () => {
    navigate("/monitor");
  };

  function handleResize() {
    if (window.innerWidth < 800) {
      setOpen(false);
      setIsLargeScreen(false);
    } else {
      setOpen(true);
      setIsLargeScreen(true);
    }
  }
  const sentiments = [
    { label: "All", icon: <CachedIcon /> },
    { label: "Positive", icon: <ThumbUpAlt /> },
    { label: "Negative", icon: <ThumbDownAlt /> }
  ];
  
  useEffect(() => {
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  useEffect(() => {
    handleResize();
  }, []);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    if (messagesEndRef?.current) {
      messagesEndRef.current.scrollTo({
        top: messagesEndRef.current.scrollHeight,
        behavior: "smooth", // Smooth scrolling
      });
    }
  };

  useEffect(() => {
    if (messages[messages.length - 1]?.sender !== "bot_rq") {
      scrollToBottom();
    }
  }, [messages, answers]);

  console.log(open, isLargeScreen);

  return (
    <>
      <div
        className="scroll-Container convogen-whole-container"
        style={{
          backgroundImage: "url('./bg.png')",
        }}
      >
        {/* sidebar */}

        {open && (
          <div
            className={
              isLargeScreen
                ? "sidebar-ui-cont hide-scrollbar large-screen"
                : "sidebar-ui-cont hide-scrollbar small-screen"
            }
            style={{
              maxWidth: drawerWidth,
            }}
          >
            <div className="sidebar-header">
              <img
                src="./infobell-large.png"
                alt="Create new chat"
                className="infobell-logo-ui"
              />
              <IconButton
                onClick={() => {
                  setOpen(false);
                }}
                style={{
                  aspectRatio: "1",
                }}
              >
                <img src="./burger.svg" alt="burger" />
              </IconButton>
            </div>
            <button
              className="btn new-chat-btn-ui"
              onClick={() => {
                window.location.reload();
              }}
            >
              + New Chat
            </button>
            <List>
              <ListItem disablePadding sx={{ marginBottom: "8px" }}>
                <ListItemButton
                  sx={{
                    "&:hover": {
                      backgroundColor: "#C7E1FE8A", // Change background color on hover
                    },
                    backgroundColor:
                      selectedItem === "uploadDoc"
                        ? "#C7E1FE8A"
                        : "transparent",
                    borderRadius: "10px",
                  }}
                  onClick={() => {
                    setMonitoring(true);
                    monitor();
                  }}
                >
                  <ListItemIcon
                    style={{ minWidth: "20px", marginRight: "8px" }}
                  >
                    <ScreenSearchDesktopIcon
                      sx={{ fontSize: 20, color: "#00A9FF" }}
                    />
                  </ListItemIcon>
                  <ListItemText primary="Monitoring" />
                </ListItemButton>
              </ListItem>
            </List>
            <List>
              <ListItem disablePadding sx={{ marginBottom: "8px" }}>
                <ListItemButton
                  sx={{
                    "&:hover": {
                      backgroundColor: "#C7E1FE8A",
                    },
                    backgroundColor:
                      selectedItem === "history" ? "#C7E1FE8A" : "transparent",
                    borderRadius: "10px",
                  }}
                  onClick={() => {
                    if (isCollapse5) setIsCollapse5(false);
                    else handleCollapse5();
                    setSelectedItem("history");
                  }}
                >
                  <ListItemIcon
                    style={{ minWidth: "20px", marginRight: "8px" }}
                  >
                    <WidgetsOutlinedIcon
                      sx={{ fontSize: 20, color: "#00A9FF" }}
                    />
                  </ListItemIcon>
                  <ListItemText primary="History" />
                  {isCollapse5 ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </ListItemButton>
              </ListItem>
              {loadingChatHistory && <p>Loading...</p>}
              <Collapse
                in={isCollapse5}
                timeout="auto"
                unmountOnExit
                sx={{
                  padding: "10px",
                }}
              >
                {files &&
                  files.map((file) => (
                    <p
                      onClick={() => {
                        setIsNewChat(false);
                        setSelectedChatHistory(file);
                        get_file({ target: { value: file } });
                      }}
                      className={`chat-history-card ${
                        selectedChatHistory === file ? "active" : ""
                      }`}
                    >
                      {file.charAt(0).toUpperCase() + file.slice(1)}
                    </p>
                  ))}
              </Collapse>
            </List>
          </div>
        )}

        <div className="scroll-Container chat-whole-container">
          {!open && (
            <IconButton className="drawer-open-icon" onClick={handleDrawerOpen}>
              <img src="./burger.svg" alt="burger" />
            </IconButton>
          )}
          <div className="amd-logo">AMD</div>

          {isNewChat ? (
            <div className="chat-content-cont">
              <div className="chat-content-header">
                <p className="chat-content-head-para">
                  Hey,{" "}
                  <i>
                    Welcome to <span>ConvoGene</span>{" "}
                  </i>
                </p>
                <p className="chat-content-title-para">How Can I Help?</p>
              </div>
              {/* Search bar */}
              <div className="big-searchBar-cont">
                <div className="big-searchBar">
                  <Button
                    className={
                      isListening
                        ? "big-search-mic-btn mic-red"
                        : "big-search-mic-btn mic-blue"
                    }
                    onClick={isListening ? stopListening : startListening}
                  >
                    {isListening ? (
                      <img
                        src={micActiveLogo}
                        alt="Mic Active"
                        style={{ height: "24px" }}
                      />
                    ) : (
                      <KeyboardVoiceIcon className="micIcon" />
                    )}
                  </Button>
                  <input
                    ref={queryRef}
                    placeholder="Enter the prompt"
                    value={searchValue}
                    onChange={handleInputChange}
                    onKeyPress={(e) => {
                      if (e.key === "Enter") getAnswer();
                    }}
                    className="bigsearhbar-input-ui"
                  />
                  <Button
                    className="big-search-send-btn"
                    onClick={() => {
                      getAnswer();
                    }}
                  >
                    <SendIcon className="sendIcon" />
                  </Button>
                </div>
              </div>
              <RandomQueries onQuerySelect={(query) => getAnswer(query)} />
            </div>
          ) : (
            <div className="d-flex flex-column align-items-center justify-content-center"
            style={{
              width: "100%", // Ensures full width from the start
              minHeight: "50px", // Stabilizes height
              marginBottom: "10px",
            }}
            >
              <div className="chat-cont">
                <div
                  ref={messagesEndRef}
                  className="scroll-Container hide-scrollbar chat-search-results-content"
                  
                >
                  {messages?.map((message, index) => (
                    <div
                      key={index}
                      className="scroll-container"
                      style={{
                        display: "flex",
                        justifyContent:
                          message?.sender === "user"
                            ? "flex-end"
                            : "flex-start",
                        flexDirection:
                          message?.sender === "user" ? "" : "column",
                      }}
                    >
                      {message?.sender === "user" && (
                        <>
                          <p
                            style={{
                              backgroundColor:
                                message?.sender === "user"
                                  ? "#C7E1FE8A"
                                  : "unset",
                            }}
                            className="wrap-text chat-wrap-text"
                          >
                            {message.answer}
                          </p>
                        </>
                      )}
                      {message?.sender === "bot" && (
                        <div className="wrap-text chat-wrap-text-bot"
                        >
                          {message?.answer && (
                            <div>
                              <img alt="" src="./star.svg" />
                            </div>
                          )}
                          {message?.isProcessing && <Shimmer />}
                          {message?.answer && (
                            <div
                              onClick={() => {
                                setMessages((prev) => {
                                  let values = [...prev];
                                  values[index]["display"] = 1;
                                  return values;
                                });
                              }}
                              style={{
                                width: message.display !== 0 ? "100%" : widthM,
                                marginBottom: "0px",
                              }}
                            >
                              <Markdown
                                remarkPlugins={[remarkGfm]}
                                components={{
                                  a: ({ node, ...props }) => (
                                    <a
                                      {...props}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                    />
                                  ),
                                }}
                              >
                                {message.answer}
                              </Markdown>
                            </div>
                          )}
                        </div>
                      )}
                      {message?.sender === "bot_rq" && (
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            p: 4,
                            margin: "5px 15px 15px 24px", // 10px for both top and bottom, 0 for left and right
                          }}
                        >
                          <ThumbUpOffAltIcon
                            sx={{
                              marginRight: "10px",
                              cursor: "pointer",
                              fill: activeIcon[`yes_${index}`]
                                ? "blue"
                                : "inherit",
                            }}
                            onClick={() => {
                              setActiveIcon((prev) => ({
                                ...prev,
                                [`yes_${index}`]: true,
                              }));
                              handleThumbClick("yes", index, {
                                question: messages[index - 2]?.answer,
                                answer: messages[index - 1]?.answer,
                              });
                            }}
                          />
                          <ThumbDownOffAltIcon
                            sx={{
                              marginRight: "10px",
                              cursor: "pointer",
                              fill: activeIcon[`no_${index}`]
                                ? "blue"
                                : "inherit",
                            }}
                            onClick={() => {
                              setActiveIcon((prev) => ({
                                ...prev,
                                [`no_${index}`]: true,
                              }));
                              handleThumbClick("no", index, {
                                question: messages[index - 2]?.answer,
                                answer: messages[index - 1]?.answer,
                              });
                            }}
                          />

                          <Typography
                            variant="body1"
                            style={{
                              fontFamily: "Inter",
                              fontSize: "15px",
                              fontWeight: 400,
                              lineHeight: "18.15px",
                              textAlign: "left",
                              textUnderlinePosition: "from-font",
                              textDecorationSkipInk: "none",
                              cursor: "pointer",
                              color: " #40444C",
                            }}
                            onClick={() =>
                              handleCommentClick(index, {
                                question: messages[index - 2]?.answer, // Previous user question
                                answer: messages[index - 1]?.answer, // Current bot answer
                              })
                            }
                          >
                            Feedback
                          </Typography>
                        </div>
                      )}

                      {message?.sender === "bot_rq" && (
                        <>
                          <div className="chat-wrap-text-bot-rq">
                            {message.answer &&
                              JSON.parse(message.answer).length > 0 && (
                                <div className="related-questions">
                                  <Typography variant="h6">
                                    Related Questions:
                                  </Typography>
                                  {JSON.parse(message.answer).map(
                                    (question, qIndex) => (
                                      <Card
                                        key={qIndex}
                                        onClick={() =>
                                          handleRelatedQuestionClick(question)
                                        }
                                        className="related-question-card"
                                      >
                                        <CardContent
                                          style={{
                                            paddingTop: "2px",
                                            paddingBottom: "2px",
                                          }}
                                        >
                                          <Box
                                            display="flex"
                                            justifyContent="space-between"
                                            alignItems="center"
                                          >
                                            <Typography variant="body1">
                                              {question}
                                            </Typography>
                                            <IconButton>
                                              <AddIcon />
                                            </IconButton>
                                          </Box>
                                        </CardContent>
                                      </Card>
                                    )
                                  )}
                                </div>
                              )}
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {Historyfeed && (
  <div className="history_feed ">
    {sentiments.map(({ label, icon }) => (
      <div
        key={label}
        className="history_thubup"
        onClick={() => handlehistoryfeed(label)}
        style={{
          backgroundColor: selectedSentiment === label ? "#5391F6" : "white",
          color: selectedSentiment === label ? "white" : "#40444C",
        }}
      >
        {icon}
        <p>{label} {label !== "All" && "Response"}</p>
      </div>
    ))}
  </div>
)}

            </div>
          )}

          {/* Search bar */}
          {!isNewChat && (
            <div className="chat-search-sub-cont">
              <div className="chat-searchbar">
                <Button
                  className={
                    isListening
                      ? "mic-search-btn mic-red"
                      : "mic-search-btn mic-blue"
                  }
                  onClick={isListening ? stopListening : startListening}
                >
                  {isListening ? (
                    <img
                      src={micActiveLogo}
                      alt="Mic Active"
                      style={{ height: "24px" }}
                    />
                  ) : (
                    <KeyboardVoiceIcon className="micIcon" />
                  )}
                </Button>
                <input
                  ref={queryRef}
                  className="searchInput search-input-ui"
                  placeholder="Enter the prompt"
                  value={searchValue}
                  onChange={handleInputChange}
                  onKeyPress={(e) => {
                    if (e.key === "Enter") getAnswer();
                  }}
                />
                <Button
                  className="search-send-btn"
                  onClick={() => {
                    getAnswer();
                  }}
                >
                  <SendIcon className="sendIcon" />
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
      <Modal open={isModalOpen} onClose={closeModal}>
        <Box
          sx={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            minWidth: "438px",
            height: "auto",
            minWidth: "400px",
            backgroundColor: " #F2F6FA",
            borderRadius: "13px ", // Updated border radius
            p: 1,
            opacity: 1,
            gap: "0px", // Added gap
            border: "0px,transparent",
          }}
        >
          <Typography
            variant="h6"
            className="comment-modal-title"
            component="h2"
            sx={{
              textAlign: "center", // Align text in the center
              display: "flex",
              justifyContent: "center", // Ensure horizontal alignment
              alignItems: "center", // Ensure vertical alignment if needed
            }}
          >
            Add Feedback
          </Typography>
          <p className="comment-modal-description">
            Add your valuable Feedback
          </p>
          <form onSubmit={handlefeedback}>
            <textarea
              id="comment-modal-description"
              style={{
                width: "100%",
                height: "auto",
                minHeight: "150px",
                marginBottom: "10px",
                border: "1px solid #5391F6",
              }}
              onChange={(e) =>
                setFeedback((prev) => ({
                  ...prev,
                  comment: e.target.value,
                }))
              }
            ></textarea>
            <div style={{ display: "flex", justifyContent: "flex-end" }}>
              <Button
                variant="contained"
                type="submit"
                mr={2}
                sx={{
                  backgroundColor: "#5391F6",
                }}
              >
                Submit Feedback
              </Button>
            </div>
          </form>
        </Box>
      </Modal>
    </>
  );
};

export default Home;
