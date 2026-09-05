# Chat formatting

Bragi uses these conventions in the chronicle:

| Content | Convention | Composer shortcut |
| --- | --- | --- |
| Narration and actions | `*I sit at the piano.*` | Alt+N |
| Spoken dialogue | `"Listen to this."` | Alt+Q |
| Phone texts | `> Meet me after the show.` | Alt+M |
| Song lyrics | A fenced block labeled `lyrics` | Alt+L |

## Lyrics

Use triple backticks followed by the lowercase word `lyrics`, then put the song
words on subsequent lines. Close the block with triple backticks on their own
line. Line breaks, blank lines between stanzas, and punctuation are preserved;
Markdown inside the verse is displayed literally.

````text
*I sit at the piano and sing quietly.*

```lyrics
I left a light beside the door,
In case your footsteps found it once more.

The evening waits upon the stair.
```

*I let the final chord fade.*
````

Select the verse and click the **Lyrics** music-note button, or press **Alt+L**
while the message field is focused. The action wraps the selected lines; with
no selection, it wraps the current line or inserts an empty block. Use it again
inside a lyrics block to remove the fences. **Clear roleplay formatting**
(**Alt+0**) also removes lyrics fences while preserving the verse's punctuation.
Selections crossing a lyrics boundary include the whole block when removing
fences, so a partial selection cannot leave an unmatched fence behind.

Surrounding narration establishes the singer, delivery, audience, and whether
the character performs, writes, or quotes the song. State whether you sing an
excerpt, pause, or finish when that distinction matters. In storyteller mode,
the message remains story direction rather than proof of a completed event.

The narrator is instructed to consider the song's established significance and
what listeners can perceive. Private intent does not become automatic audience
knowledge. Lyrical imagery does not establish literal events, biography,
promises, or world changes. Summaries and context maintenance receive the same
guidance to retain the performance and supported significance without treating
the lyrics as factual claims. The narrator must preserve player agency and
leave unsupplied player lyrics and unexpressed intentions to the player.

These are interpretation instructions for the model; the formatting itself does
not guarantee a particular character reaction. Lyrics remain ordinary message
text in saves and chat exports, so they require no migration or special import
format.
