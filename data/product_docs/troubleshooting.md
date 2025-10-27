# Troubleshooting Guide

## Common Issues and Solutions

### Login and Authentication Issues

#### Problem: "Invalid email or password" error

**Solution:**
1. Verify you're using the correct email address
2. Click **"Forgot Password"** to reset your password
3. Check if Caps Lock is enabled
4. Clear browser cache and cookies
5. Try a different browser or incognito mode

#### Problem: Not receiving password reset email

**Solution:**
1. Check your spam/junk folder
2. Ensure support@taskflowpro.com is not blocked
3. Wait 5-10 minutes (emails can be delayed)
4. Try requesting another reset link
5. Contact support if still not received after 30 minutes

#### Problem: Two-Factor Authentication (2FA) code not working

**Solution:**
1. Ensure your device clock is synchronized
2. Try the next code (codes refresh every 30 seconds)
3. Use backup codes if available (Settings → Security)
4. Contact support to temporarily disable 2FA: support@taskflowpro.com

### Performance Issues

#### Problem: TaskFlow Pro is loading slowly

**Solution:**
1. Check your internet connection speed
2. Close unnecessary browser tabs
3. Clear browser cache:
   - Chrome: Ctrl+Shift+Delete (Cmd+Shift+Delete on Mac)
   - Firefox: Ctrl+Shift+Delete
   - Safari: Cmd+Option+E
4. Disable browser extensions temporarily
5. Try using a different browser
6. Check our status page: status.taskflowpro.com

#### Problem: Tasks not updating in real-time

**Solution:**
1. Refresh the page (F5 or Cmd+R)
2. Check if you're connected to the internet
3. Verify you have the latest app version (web updates automatically)
4. Close and reopen the application
5. Try accessing from a different device

### Task Management Issues

#### Problem: Cannot create new tasks

**Solution:**
1. **Check permissions**: Ensure you have "Member" or "Admin" role (not "Viewer")
2. **Project limit**: Free plan limited to 3 projects - upgrade or delete old projects
3. **Browser issues**: Try refreshing the page or clearing cache
4. **Network**: Verify internet connection
5. Contact support if error persists

#### Problem: Tasks disappearing or not saving

**Solution:**
1. Check if tasks were moved to archive
2. Verify you're in the correct project
3. Check filters (tasks might be hidden by active filters)
4. Look in **"Recently Deleted"** (Settings → Data Recovery)
5. Tasks can be restored within 30 days of deletion

#### Problem: Cannot upload attachments

**Possible causes and solutions:**

**File too large**
- Maximum file size: 50MB per file
- Compress large files or use cloud storage links

**Unsupported file type**
- Blocked types: .exe, .bat, .com (security reasons)
- Use ZIP compression for these file types

**Storage quota exceeded**
- Free: 1GB total
- Professional: 100GB per user
- Check storage usage: Settings → Storage
- Delete unnecessary files or upgrade plan

### Integration Issues

#### Problem: Slack integration not working

**Solution:**
1. Reconnect Slack: Settings → Integrations → Slack → Reconnect
2. Verify Slack workspace permissions
3. Ensure TaskFlow Pro bot is added to relevant channels
4. Check notification settings in both Slack and TaskFlow Pro
5. Re-authorize the integration

#### Problem: Calendar sync not updating

**Solution:**
1. Check sync settings: Settings → Integrations → Calendar
2. Verify calendar permissions (read/write access required)
3. Force manual sync: Click "Sync Now" button
4. Check if calendar is subscribed (not just added)
5. Disconnect and reconnect calendar

### Mobile App Issues

#### Problem: Mobile app not syncing

**Solution:**
1. Pull down to refresh (swipe down on main screen)
2. Check internet connection (WiFi or cellular data)
3. Verify you're logged into the correct account
4. Force close and reopen app
5. Update to latest app version
6. Reinstall the app (data is stored in cloud)

#### Problem: Push notifications not working

**Solution:**
1. Check notification permissions:
   - iOS: Settings → TaskFlow Pro → Notifications
   - Android: Settings → Apps → TaskFlow Pro → Notifications
2. Verify notification settings in app: Settings → Notifications
3. Restart device
4. Reinstall the app

### Collaboration Issues

#### Problem: Team members not receiving invitations

**Solution:**
1. Verify email addresses are correct
2. Check their spam/junk folder
3. Resend invitation: Team → Members → Resend
4. Ensure your plan supports additional members:
   - Free: Up to 5 members
   - Professional: Unlimited
5. Use direct invite link: Team → Get Invite Link

#### Problem: Cannot mention team members in comments

**Solution:**
1. Ensure user is a project member
2. Type @ symbol followed by name
3. Wait for autocomplete dropdown (may take 1-2 seconds)
4. Select from dropdown list
5. Check if user account is active (not deactivated)

### Data and Export Issues

#### Problem: Cannot export data

**Solution:**
1. **Check permissions**: Only Admins can export
2. **Large projects**: Export may take several minutes
3. **Format**: Choose CSV, Excel, or JSON format
4. **Browser**: Some browsers block downloads - check popup blocker
5. **Try**: Settings → Data Export → Select what to export

#### Problem: Imported data not showing correctly

**Solution:**
1. Verify CSV format matches template (download sample template)
2. Check for special characters or encoding issues (use UTF-8)
3. Required fields must be filled
4. Date format: YYYY-MM-DD
5. Review import log for errors: Settings → Import History

## Error Messages

### "Something went wrong"
- Generic error - usually temporary
- Refresh the page
- If persistent, contact support with error code

### "You don't have permission"
- Your role (Viewer) doesn't allow this action
- Request Admin or Member role from project owner

### "Project limit reached"
- Free plan: 3 projects maximum
- Upgrade to Professional for unlimited projects
- Or delete/archive old projects

### "Session expired"
- You've been logged out for security
- Log in again
- Enable "Remember me" for longer sessions

### "Rate limit exceeded"
- Too many requests in short time
- Wait 60 seconds and try again
- Usually happens with API or bulk operations

## Browser Compatibility

### Supported Browsers

**Fully Supported:**
- Chrome/Edge (version 90+)
- Firefox (version 88+)
- Safari (version 14+)

**Not Supported:**
- Internet Explorer (discontinued)
- Opera Mini
- Browsers older than 2 years

### Recommended Settings

- Enable JavaScript (required)
- Enable cookies (required)
- Disable ad blockers for taskflowpro.com
- Allow popups for taskflowpro.com
- Enable local storage

## System Status and Maintenance

Check our system status: **status.taskflowpro.com**

**Planned Maintenance:**
- Scheduled Saturdays 2am-4am EST
- Announced 48 hours in advance
- Updates posted on status page

## Getting Additional Help

### Before Contacting Support

1. Check this troubleshooting guide
2. Visit our Help Center: help.taskflowpro.com
3. Search Community Forum: community.taskflowpro.com
4. Check status page: status.taskflowpro.com

### Contact Support

**Email Support**
- Address: support@taskflowpro.com
- Response time: 
  - Free: Within 48 hours
  - Professional: Within 24 hours
  - Enterprise: Within 4 hours

**Live Chat**
- Available: Monday-Friday, 9am-6pm EST
- Click chat icon in bottom right of screen
- Professional and Enterprise plans only

**Phone Support**
- Enterprise customers only
- Call your dedicated account manager

### Information to Include

When contacting support, please provide:

1. **Account email address**
2. **Browser and version** (e.g., Chrome 120)
3. **Operating System** (e.g., Windows 11, MacOS 14)
4. **Steps to reproduce** the issue
5. **Screenshots or screen recordings** if possible
6. **Error messages** (exact text or screenshot)
7. **When did the issue start?**

## Known Issues

Current known issues are tracked at: status.taskflowpro.com/known-issues

We regularly update this page with workarounds and expected fix dates.

## Feature Requests

Have an idea for improving TaskFlow Pro?

1. Visit: feedback.taskflowpro.com
2. Search existing requests
3. Vote on existing ideas or submit new ones
4. Track implementation status

Most requested features are prioritized in our roadmap!

