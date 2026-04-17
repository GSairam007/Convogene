import "./App.css";
import { ThemeProvider, createTheme } from "@mui/material";
import "bootstrap/dist/css/bootstrap.min.css";
import { BrowserRouter } from "react-router-dom";
import Router from "./routes";
import { Provider } from "react-redux";
import store from "./store";

const theme = createTheme({
  palette: {
    primary: {
      main: "#9F2AEC", // Your custom primary color
    },
    secondary: {
      main: "#6c757d",
    },
    info: {
      main: "#000", // Your custom primary color
    },
    tertiary: {
      main: "#6c757d",
    },
    // You can also customize other palette colors here
  },
});
function App() {
  return (
    <>
      <ThemeProvider theme={theme}>
        <Provider store={store}>
          <BrowserRouter>
            <Router />
          </BrowserRouter>
        </Provider>
      </ThemeProvider>
    </>
  );
}

export default App;
