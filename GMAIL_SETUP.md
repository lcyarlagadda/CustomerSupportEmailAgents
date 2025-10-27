# Gmail Integration Setup

Your system is now configured to use **real Gmail** instead of mock emails! 

## What I Did

✅ Fixed filename: `creditials.json` → `credentials.json`  
✅ Updated `utils/email_handler.py` to use Gmail API  
✅ Installed Gmail API packages  
✅ System now reads and sends real emails  

## Your credentials.json

```json
{
  "web": {
    "client_id": "24672502881-...",
    "project_id": "emailsupport-476415",
    ...
  }
}
```

✅ Located at: `credentials.json`  
✅ Ready to use!

## How to Run

### First Time Setup

1. **Set your Gmail address** (the one you'll monitor):

Create `.env` file:
```bash
GMAIL_EMAIL=your.email@gmail.com
HF_TOKEN=your_hf_token_here
```

2. **Run the system**:

```bash
python main.py --mode batch
```

3. **Authenticate** (first time only):
   - Browser will automatically open
   - Sign in with your Gmail account
   - Click "Allow" to grant permissions
   - Browser will show "Authentication successful"
   - Close browser and return to terminal

4. **Authentication saved**:
   - Token saved to `token.json`
   - Future runs won't need authentication
   - Token auto-refreshes when expired

### Subsequent Runs

After first authentication:

```bash
python main.py --mode batch
```

No browser needed - uses saved `token.json`!

## What the System Does

### 1. Check for Unread Emails

Monitors your Gmail inbox for unread emails:
- Fetches up to 10 unread emails
- Extracts sender, subject, body
- Processes through AI workflow

### 2. Process Each Email

- Classifies category and priority
- Retrieves relevant product documentation
- Generates personalized response
- Quality checks the response

### 3. Send Replies

- Sends response via Gmail
- Maintains email thread
- Marks original email as read
- Logs all activity

## Modes

### Batch Mode (Recommended)

Process all current unread emails once:

```bash
python main.py --mode batch
```

### Continuous Mode

Monitor inbox continuously:

```bash
python main.py --mode continuous
```

- Checks every 60 seconds (configurable in `utils/config.py`)
- Press Ctrl+C to stop
- Processes new emails as they arrive

## Testing

### Test with Your Own Email

1. Send an email TO your Gmail address
2. Email should be unread in inbox
3. Run the system:
```bash
python main.py --mode batch
```

4. System will:
   - Find the unread email
   - Process it through AI agents
   - Generate and send response
   - Mark as read

### Example Test Email

Send yourself an email:

**Subject**: "How do I create a project?"

**Body**:
```
Hi Support,

I'm new to TaskFlow Pro and I'd like to know how to create
my first project. Can you help?

Thanks!
```

System will generate an appropriate response!

## Gmail API Permissions

The system has these permissions:
- ✅ Read emails
- ✅ Send emails
- ✅ Modify labels (mark as read)
- ✅ Maintain threads

**Scopes**: `https://www.googleapis.com/auth/gmail.modify`

## Configuration Options

### In `.env` file:

```bash
# Your Gmail address (required)
GMAIL_EMAIL=your.actual.email@gmail.com

# HuggingFace token (for Llama)
HF_TOKEN=hf_your_token_here

# LLM Model (optional)
LLM_MODEL=meta-llama/Llama-3.2-1B-Instruct

# Check interval for continuous mode (optional)
# Edit in utils/config.py: CHECK_INTERVAL_SECONDS = 60
```

## File Locations

```
ProductSupportAgents/
├── credentials.json       ✅ OAuth credentials (your file)
├── token.json            📝 Created after first auth
├── .env                  📝 Create this (your settings)
├── utils/
│   └── email_handler.py  ✅ Updated to use Gmail API
└── logs/                 📝 Email processing logs
```

## Troubleshooting

### "Credentials not found"

Make sure `credentials.json` exists in project root:
```bash
ls credentials.json  # Should exist
```

### "Access blocked" during authentication

Your Google Cloud project needs:
1. Gmail API enabled (should already be done)
2. OAuth consent screen configured
3. Your email added as test user (if app is in testing mode)

### "Token expired"

Delete `token.json` and re-authenticate:
```bash
rm token.json
python main.py
```

### "No emails found"

- Check Gmail inbox has unread emails
- Verify GMAIL_EMAIL in .env matches your Gmail
- Try sending yourself a test email

### "Permission denied"

Re-authenticate with correct account:
```bash
rm token.json
python main.py
```

## Security Notes

### Keep These Private

**Never share or commit:**
- ❌ `credentials.json` - OAuth credentials
- ❌ `token.json` - Access token
- ❌ `.env` - Your settings

Already in `.gitignore` ✅

### Token Security

- `token.json` contains your access token
- Auto-refreshes when expired
- Revoke access anytime in [Google Account](https://myaccount.google.com/permissions)

## Advanced Usage

### Custom Email Filters

Modify query in `utils/email_handler.py`:

```python
# Currently fetches all unread
q='is:unread in:inbox'

# Examples:
q='is:unread subject:support'           # Only support emails
q='is:unread from:customer@company.com' # From specific sender
q='is:unread label:urgent'              # With specific label
```

### Increase Email Limit

In `utils/email_handler.py`:

```python
maxResults=10  # Change to 50, 100, etc.
```

### Custom Scopes

Edit `utils/email_handler.py`:

```python
# Current (modify only)
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# Full access
SCOPES = ['https://www.googleapis.com/auth/gmail.full']

# Read only (testing)
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
```

After changing scopes, delete `token.json` and re-authenticate.

## Production Deployment

### For Production Use

1. **Move OAuth app to production** (Google Cloud Console)
2. **Add all user emails** or make app public
3. **Set up monitoring** for errors
4. **Configure email filters** to avoid spam
5. **Set rate limits** to avoid Gmail API quotas
6. **Add error notifications**

### Gmail API Quotas

Free tier limits:
- 1 billion requests/day
- More than enough for most use cases

## Next Steps

1. ✅ Authentication working? Test with real email
2. ✅ Responses good? Customize prompts in `agents/`
3. ✅ Ready for more emails? Use continuous mode
4. ✅ Going to production? Follow deployment guide

## Support

If you encounter issues:
1. Check `logs/` directory for error messages
2. Verify `credentials.json` and `token.json` exist
3. Ensure Gmail API is enabled in Google Cloud
4. Try deleting `token.json` and re-authenticating

---

**Your system is now connected to real Gmail! 🎉**

Run `python main.py --mode batch` to process your first real email!

