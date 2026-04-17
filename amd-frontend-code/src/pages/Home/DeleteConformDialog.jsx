
import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { LoadingButton } from '@mui/lab';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
} from '@mui/material';

const DeleteConformDialog = ({
  open,
  onClose,
  onConfirm,
  itemName,
  message,
  confirmButtonText,
  cancelButtonText,
  isLoadingRequired,
  confirmButtonColor,
}) => {
  const [isLoading, setLoading] = useState(true);
  const [countdown, setCountdown] = useState(3);

  const startLoading = () => {
    const interval = setInterval(() => {
      setCountdown((prevCount) => {
        const newCount = prevCount - 1;
        if (newCount <= 0) {
          clearInterval(interval);
          setLoading(false);
        }
        return newCount;
      });
    }, 1000);
  };

  useEffect(() => {
    if (open) {
      setCountdown(3); // Reset countdown when the dialog is opened
      setLoading(true);
      startLoading();
    }
  }, [open]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle>{itemName}</DialogTitle>
      <DialogContent>
        <DialogContentText variant="subtitle1">{message}</DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} variant="outlined" color="primary">
          {cancelButtonText || 'Cancel'}
        </Button>
        {isLoading && isLoadingRequired ? (
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <LoadingButton
              size="small"
              loading
              variant="outlined"
              color="primary"
              style={{ marginLeft: '10px' }}
            >
              Verify
            </LoadingButton>
            <p style={{ marginLeft: '10px' }}>{countdown}s</p>
          </div>
        ) : (
          <Button onClick={onConfirm} variant="contained" color={confirmButtonColor || 'error'}>
            {confirmButtonText || 'Delete'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

DeleteConformDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
  onConfirm: PropTypes.func.isRequired,
  itemName: PropTypes.string.isRequired,
  message: PropTypes.string.isRequired,
  confirmButtonText: PropTypes.string,
  cancelButtonText: PropTypes.string,
};

export default DeleteConformDialog;