import React, { useState } from "react";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TablePagination from "@mui/material/TablePagination";
import TableRow from "@mui/material/TableRow";
import Popover from "@mui/material/Popover";
import Typography from "@mui/material/Typography";
import ReactMarkdown from "react-markdown";
import { Box, Button, Modal } from "@mui/material";
import Grid from "@mui/material/Grid";

const columns = [
  { id: "Question", label: "Question", minWidth: 50 },
  { id: "Name", label: "Name", minWidth: 30 },
  { id: "Time", label: "Time", minWidth: 50 },
  { id: "Total Tokens", label: "Total Tokens", minWidth: 50 },
  { id: "Total Cost($)", label: "Total Cost ($)", minWidth: 50 },
  { id: "Total Cost(rs)", label: "Total Cost (₹)", minWidth: 50 },
];

export default function ColumnGroupingTable({ tableData }) {
  console.log(tableData, "table data");
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [anchorEl, setAnchorEl] = useState(null);
  const [PopDat, setPopDat] = useState({});
  const [showPreviewModal, setShowPreviewModal] = useState(false);

  const [popoverContent, setPopoverContent] = useState({
    input: "",
    output: "",
  });

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  function handleModalClose() {
    setShowPreviewModal(false);
  }
  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(+event.target.value);
    setPage(0);
  };

  const handlePopoverOpen = (event, row) => {
    console.log(row, "row data ");
    setShowPreviewModal(true);
    setPopDat(row);
    setAnchorEl(event.currentTarget);

    // Convert input and output to strings if they are objects
    const inputContent = JSON.stringify(row.Inputs, null, 2); // Pretty-prints JSON
    const outputContent = JSON.stringify(row.outputs, null, 2);

    setPopoverContent({ input: inputContent, output: outputContent });
  };

  const handlePopoverClose = () => {
    setAnchorEl(null);
  };

  const open = Boolean(anchorEl);

  return (
    <>
      <Paper sx={{ width: "100%", marginTop: "20px" }} className="custom-paper">
        <TableContainer
          sx={{ maxHeight: 440, overflowY: "auto" }}
          className="custom-paper"
        >
          {" "}
          {/* Allow the table to scroll */}
          <Table
            stickyHeader
            aria-label="sticky table"
            className="custom-paper"
          >
            <TableHead>
              <TableRow className="custom-tableHead">
                {columns.map((column) => (
                  <TableCell
                    key={column.id}
                    align={column.align}
                    style={{ minWidth: column.minWidth }}
                    className="css-y8ay40-MuiTableCell-root"
                    // sx={{ fontSize: "0.7rem" }} // Reduced font size
                  >
                    {column.label}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {tableData
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((row) => {
                  return (
                    <TableRow
                      hover
                      role="checkbox"
                      tabIndex={-1}
                      key={row.ID}
                      onClick={(event) => handlePopoverOpen(event, row)}
                      // onMouseEnter={(event) => handlePopoverOpen(event, row)}
                      onMouseLeave={handlePopoverClose}
                    >
                      {columns.map((column) => {
                        const value = row[column.id];
                        return (
                          <TableCell
                            key={column.id}
                            align={column.align}
                            sx={{ fontSize: "1rem" }}
                          >
                            {typeof value === "object" ? (
                              <pre>{JSON.stringify(value, null, 2)}</pre>
                            ) : (
                              value
                            )}
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  );
                })}
            </TableBody>
          </Table>
        </TableContainer>

        <TablePagination
          rowsPerPageOptions={[10, 25, 100]}
          component="div"
          count={tableData.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
        <Popover
          id="mouse-over-popover"
          sx={{
            pointerEvents: "none",
            "& .MuiPopover-paper": {
              padding: "16px",
              maxWidth: "75%",
              maxHeight: "60%",
              overflow: "hidden", // Prevent the popover from scrolling out
            },
          }}
          open={open}
          anchorEl={anchorEl}
          anchorOrigin={{
            vertical: "bottom",
            horizontal: "left",
          }}
          transformOrigin={{
            vertical: "top",
            horizontal: "left",
          }}
          onClose={handlePopoverClose}
          disableRestoreFocus
        >
          <div style={{ maxHeight: "200px", overflowY: "auto" }}>
            {/* Set max height for scrolling */}
            <Typography variant="body1" gutterBottom>
              Input:
            </Typography>
            <Typography fontSize="0.5rem">
              <ReactMarkdown>{popoverContent.input}</ReactMarkdown>
            </Typography>
            <Typography variant="body1" gutterBottom>
              Output:
            </Typography>
            <Typography fontSize="0.5rem">
              <ReactMarkdown>{popoverContent.output}</ReactMarkdown>
            </Typography>
          </div>
        </Popover>
      </Paper>

      <Modal
        open={showPreviewModal}
        className="modal-style"
        onClose={handleModalClose}
        aria-labelledby="question-details-title"
        aria-describedby="question-details-description"
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Box
          sx={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            bgcolor: "background.paper",
            boxShadow: 24,
            borderRadius: 2, // Increase border radius for aesthetics
          }}
          className="dash-popup"
        >
          <Box
            display="flex"
            justifyContent="center"
            alignItems="center"
            mb={4}
            position="relative"
          >
            <Typography
              id="question-details-title"
              variant="h5"
              component="h3"
              fontWeight="bold"
              sx={{ fontSize: "1.5rem", textAlign: "center", width: "100%" }}
            >
              Question Details
            </Typography>
            <Button
              onClick={handleModalClose}
              sx={{
                fontSize: "1.5rem",
                minWidth: "unset",
                position: "absolute",
                right: 0,
              }}
            >
              &times;
            </Button>
          </Box>

          {PopDat ? (
            <Box display="flex" flexDirection="column" gap={2} className="common-modal-grid-ui">
              <Box display="flex">
                <Grid className="modal-grid-item" item xs={6} md={4}>
                  <p className="dash-pop-text-bold">Name <span className="popup-colon">:</span> </p>
                </Grid>
                <Grid className="modal-grid-item" item xs={6} md={8}>
                  <p className="dash-pop-text">{PopDat.Name || "N/A"}</p>
                </Grid>
              </Box>
              <Box display="flex">
                <Grid className="modal-grid-item" item xs={6} md={4}>
                  <p className="dash-pop-text-bold">Question <span className="popup-colon">:</span>  </p>
                </Grid>
                <Grid className="modal-grid-item" item xs={6} md={8}>
                  <p className="dash-pop-text">{PopDat.Question || "N/A"}</p>
                </Grid>
              </Box>
              <Box display="flex">
                <Grid className="modal-grid-item" item xs={6} md={4}>
                  <p className="dash-pop-text-bold">Total Cost($) <span className="popup-colon">:</span>  </p>
                </Grid>
                <Grid className="modal-grid-item" item xs={6} md={8}>
                  <p className="dash-pop-text">
                    {PopDat?.["Total Cost($)"] || "N/A"}
                  </p>
                </Grid>
              </Box>
              <Box display="flex">
                <Grid className="modal-grid-item" item xs={6} md={4}>
                  <p className="dash-pop-text-bold">Total Cost (₹) <span className="popup-colon">:</span>  </p>
                </Grid>
                <Grid className="modal-grid-item" item xs={6} md={8}>
                  <p className="dash-pop-text">
                    {PopDat?.["Total Cost(rs)"] || "N/A"}
                  </p>
                </Grid>
              </Box>
              <Box display="flex">
                <Grid className="modal-grid-item" item xs={6} md={4}>
                  <p className="dash-pop-text-bold">Total Tokens <span className="popup-colon">:</span> </p>
                </Grid>
                <Grid className="modal-grid-item" item xs={6} md={8}>
                  <p className="dash-pop-text">
                    {PopDat?.["Total Tokens"] || "N/A"}
                  </p>
                </Grid>
              </Box>
              <Box display="flex">
                <Grid className="modal-grid-item" item xs={6} md={4}>
                  <p className="dash-pop-text-bold">Time <span className="popup-colon">:</span>  </p>
                </Grid>
                <Grid className="modal-grid-item" item xs={6} md={8}>
                  <p className="dash-pop-text">{PopDat?.["Time"] || "N/A"}</p>
                </Grid>
              </Box>
            </Box>
          ) : (
            <Typography variant="body2" sx={{ fontSize: "1.2rem" }}>
              No details available.
            </Typography>
          )}
        </Box>
      </Modal>
    </>
  );
}
