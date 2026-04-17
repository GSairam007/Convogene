// dataActions.js
export const setData = (data) => {
    console.log(data +"data print")
    return {
        type: 'SET_DATA',
        recordedVideos: data,
    };
};

// dataReducer.js
const initialState = {
    recordedVideos: [],
};

const dataReducer = (state = initialState, action) => {
//    This place might the place to set data
    switch (action.type) {
        case 'SET_DATA':
            return { ...state, recordedVideos: action.recordedVideos };
        default:
            return state;
    }
};

export default dataReducer;
