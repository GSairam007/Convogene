import styles from './iconHolder.module.css'
import HomeIcon from '@mui/icons-material/Home';
// import GroupsIcon from '@mui/icons-material/Groups';
import SubscriptionsIcon from '@mui/icons-material/Subscriptions';
// import CalendarMonthIcon from '@mui/icons-material/CalendarMonth';
// import ArticleIcon from '@mui/icons-material/Article';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronRightIcon from '@mui/icons-material/ChevronRight'; 
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
// import MicIcon from '@mui/icons-material/Mic';
import RobotIcon from './../../assets/icons/robot.svg'
import ZoomMeetingIcon from './../../assets/icons/zoom-icon.svg'
import DocumentIcon from './../../assets/icons/icons8-document.svg'
import GoogleMeetingIcon from './../../assets/icons/google-meet-icon.svg'
import WebexMeetingIcon from './../../assets/icons/webex-logo.svg'
import MicrosoftTeamsIcon from './../../assets/icons/microsoft-teams.svg';
import YoutubeIcon from './../../assets/icons/youtube.svg'
import Video1Icon from './../../assets/icons/video1.svg';
import Music1Icon from './../../assets/icons/music1.svg';
import Img1 from './../../assets/icons/img.svg';
import Text1 from './../../assets/icons/text.svg';
import CloseIcon from '@mui/icons-material/Close';
import OndemandVideoIcon from '@mui/icons-material/OndemandVideo';
import DeleteIcon from '@mui/icons-material/Delete';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import DoneIcon from '@mui/icons-material/Done';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import RecordVoiceOverIcon from '@mui/icons-material/RecordVoiceOver';
import VideoSettingsIcon from '@mui/icons-material/VideoSettings';
import PhotoIcon from '@mui/icons-material/Photo';
import SendIcon from '@mui/icons-material/Send';
import InfobellConvo from "../../assets/Images/infobellLogo.png"
import Woman from "../../assets/Images/icons8-woman-head-50.png"
import Plane from "../../assets/Images/icons8-plane-48.png"
import Graduate from "../../assets/Images/icons8-graduate-cap-96.png"
import Writer from "../../assets/Images/icons8-writer-50.png"
import Security from "../../assets/Images/icons8-security-lock-100.png"
import Paperplane from "../../assets/Images/paperIcon.png"

const IconsHolder = (props) => {
    if(props.type === 'home'){
        return (<HomeIcon style={{ color: props.color ? props.color : '' }} />)
    } else if(props.type === 'subscription'){
        return (<SubscriptionsIcon style={{ color: props.color ? props.color : '' }} />)
    } 
    
    else if (props.type === 'menu'){
        return (<MenuIcon style={{ color: props.color ? props.color : '' }} />)
    } else if (props.type === 'ChevronRightIcon'){
        return (<ChevronRightIcon style={{ color: props.color ? props.color : '' }} />)
    } else if (props.type === 'ChevronLeftIcon'){
        return (<ChevronLeftIcon style={{ color: props.color ? props.color : '' }} />)
    }  else if (props.type === 'CloseIcon'){
        return (<CloseIcon style={{ color: props.color ? props.color : '' }} />)
    } else if (props.type === 'OndemandVideoIcon'){
        return (<OndemandVideoIcon style={{ color: props.color ? props.color : '' }} />)
    } else if (props.type === 'DeleteIcon'){
        return (<DeleteIcon style={{ color: props.color ? props.color : '' }} onClick={() => { props.onClick() }} />)
    }
    else if (props.type === 'DoneIcon'){
        return (<DoneIcon style={{ color: props.color ? props.color : '' }} />)
    } else if (props.type === 'MoreVertIcon'){
        return (<MoreVertIcon style={{ color: props.color ? props.color : '' }} />)
    } else if (props.type === 'UploadFileIcon'){
        return (<UploadFileIcon style={{ color: props.color ? props.color : '' }} />)
    } 
    else if (props.type === 'RecordVoiceOverIcon'){
        return (<RecordVoiceOverIcon style={{ color: props.color ? props.color : '' }} />)
    }
    else if (props.type === 'PhotoIcon'){
        return (<PhotoIcon style={{ color: props.color ? props.color : '' }} />)
    }
    else if (props.type === 'VideoSettingsIcon'){
        return (<VideoSettingsIcon style={{ color: props.color ? props.color : '' }} />)
    }
    else if (props.type === 'SendIcon'){
        return (<SendIcon style={{ color: props.color ? props.color : '' }} />)
        }
    else if (props.type === 'ZoomMeetingIcon'){
        return (<img className={styles.imageIcon} src={ZoomMeetingIcon} alt='Zoom'/>)
    } else if (props.type === 'googleMeetIcon'){
        return (<img className={styles.imageIcon} src={GoogleMeetingIcon} alt='Zoom'/>)
    } 
    else if (props.type === 'Woman'){
        return (<img className={styles.imageIcon} src={Woman} alt='Zoom'/>)
    } 
    else if (props.type === 'Security'){
        return (<img className={styles.imageIcon} src={Security} alt='Zoom'/>)
    } 

    else if (props.type === 'Writer'){
        return (<img className={styles.imageIcon} src={Writer} alt='Zoom'/>)
    } 
    else if (props.type === 'Plane'){
        return (<img className={styles.imageIcon} src={Plane} alt='Zoom'/>)
    } 
    else if (props.type === 'Paperplane'){
        return (<img className={styles.imageIcon} src={Paperplane} alt='Zoom'/>)
    } 
    else if (props.type === 'Graduate'){
        return (<img className={styles.imageIcon} src={Graduate} alt='Zoom'/>)
    } 

    else if (props.type === 'YoutubeIcon'){
        return (<img className={styles.imageIcon} src={YoutubeIcon} alt='Zoom'/>)
    }
    else if (props.type === 'Video1Icon'){
        return (<img className={styles.imageIcon} src={Video1Icon} alt='Zoom'/>)
    }
    else if (props.type === 'Music1Icon'){
        return (<img className={styles.imageIcon} src={Music1Icon} alt='Zoom'/>)
    }

    

    else if (props.type === 'MicrosoftTeamsIcon'){
        return (<img className={styles.imageIcon} src={MicrosoftTeamsIcon} alt='Zoom'/>)
    }
    else if (props.type === 'InfobellConvo'){
        return (<img className={styles.imageIcon} src={InfobellConvo} alt='Zoom'/>)
    }
    else if (props.type === 'Img1'){
        return (<img className={styles.imageIcon} src={Img1} alt='Zoom'/>)
    }
    else if (props.type === 'WebexMeetingIcon'){
            return (<img className={styles.imageIcon} src={WebexMeetingIcon} alt='Zoom'/>)
    } 
    else if (props.type === 'RobotIcon'){
        return (<img className={styles.imageIcon} src={RobotIcon} alt='Zoom'/>)
} 
    else if (props.type === 'DocumentIcon'){
        return (<img className={styles.imageIcon} src={DocumentIcon} alt='Zoom'/>)
    }
    else if (props.type === 'Text1'){
        return (<img className={styles.imageIcon} src={Text1} alt='Zoom'/>)
    }
    else {
        return (<></>)
    }

}


export default IconsHolder;
