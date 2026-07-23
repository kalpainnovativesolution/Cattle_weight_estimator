/**
 * =============================================================================
 * Cattle Weight Estimator — Google Apps Script Web App backend
 * =============================================================================
 * Paste this entire file into the Apps Script editor (Extensions > Apps
 * Script) of the Google Sheet you want to use as your log database.
 *
 * - doGet()  -> returns ALL rows in the sheet as a JSON array of objects
 *               (used by Streamlit's "View All Logs" / load_logs_from_google_sheet())
 * - doPost() -> appends ONE new row to the sheet, never overwriting existing
 *               data (used by Streamlit's append_log_to_google_sheet())
 *
 * The header row is created automatically the first time a row is appended,
 * so you do NOT need to manually type headers into the sheet beforehand.
 * =============================================================================
 */

// Column order must match Streamlit's LOG_COLUMNS exactly.
var HEADERS = [
  "Tag_ID",
  "Date",
  "Time",
  "Linear_Body_Depth_cm",
  "Linear_Chest_Height_cm",
  "Body_Length_cm",
  "Heart_Girth_cm",
  "Weight_kg",
];

// If you want to target a specific sheet/tab name, set it here.
// Leave as null to just use the first (active) sheet in the spreadsheet.
var SHEET_NAME = null; // e.g. "Logs"

function _getSheet_() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  if (SHEET_NAME) {
    var named = ss.getSheetByName(SHEET_NAME);
    if (named) return named;
  }
  return ss.getSheets()[0];
}

function _jsonOutput_(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

/**
 * GET -> return every logged row as a JSON array of objects, e.g.:
 *   [{"Tag_ID": "COW-1024", "Date": "2026-07-23", ... }, ...]
 * Returns [] if the sheet has no data rows yet.
 */
function doGet(e) {
  try {
    var sheet = _getSheet_();
    var lastRow = sheet.getLastRow();
    var lastCol = sheet.getLastColumn();

    if (lastRow < 2 || lastCol < 1) {
      // No data rows (only headers, or completely empty sheet)
      return _jsonOutput_([]);
    }

    var data = sheet.getRange(1, 1, lastRow, lastCol).getValues();
    var headers = data[0];
    var rows = [];

    for (var i = 1; i < data.length; i++) {
      var rowObj = {};
      for (var j = 0; j < headers.length; j++) {
        var value = data[i][j];
        // Normalize Date objects (e.g. if a cell was auto-formatted as a date)
        if (Object.prototype.toString.call(value) === "[object Date]") {
          value = Utilities.formatDate(value, Session.getScriptTimeZone(), "yyyy-MM-dd HH:mm:ss");
        }
        rowObj[headers[j]] = value;
      }
      rows.push(rowObj);
    }

    return _jsonOutput_(rows);
  } catch (err) {
    return _jsonOutput_({ error: err.toString() });
  }
}

/**
 * POST -> append one new row built from the JSON body, e.g.:
 *   {"Tag_ID": "COW-1024", "Date": "2026-07-23", "Time": "14:02:11",
 *    "Linear_Body_Depth_cm": 61.2, "Linear_Chest_Height_cm": 128.4,
 *    "Body_Length_cm": 142.7, "Heart_Girth_cm": 178.3, "Weight_kg": 521.9}
 * Always appends — never overwrites previous rows.
 */
function doPost(e) {
  try {
    if (!e || !e.postData || !e.postData.contents) {
      return _jsonOutput_({ status: "error", message: "No POST body received." });
    }

    var payload = JSON.parse(e.postData.contents);
    var sheet = _getSheet_();

    // Write the header row automatically if the sheet is currently empty.
    if (sheet.getLastRow() === 0) {
      sheet.appendRow(HEADERS);
    }

    var row = HEADERS.map(function (h) {
      var v = payload[h];
      return (v === undefined || v === null) ? "" : v;
    });

    sheet.appendRow(row);

    return _jsonOutput_({ status: "success" });
  } catch (err) {
    return _jsonOutput_({ status: "error", message: err.toString() });
  }
}
